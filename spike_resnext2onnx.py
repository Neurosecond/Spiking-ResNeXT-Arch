import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

import onnx
import onnxruntime as ort

# =============================================================================
# 2. КОДИРОВЩИК ДАННЫХ В СПАЙКИ
# =============================================================================

class FinancialSpikeEncoder:
    """Кодировщик финансовых данных в спайки"""

    def __init__(self, num_time_steps=50, threshold_std=1.0):
        self.num_time_steps = num_time_steps
        self.threshold_std = threshold_std

    def price_change_encoding(self, price_data):
        """Кодирование изменений цены в спайки"""
        # price_data shape: (batch_size, seq_len, num_features)
        batch_size, seq_len, num_features = price_data.shape

        # Вычисляем изменения цены для каждого признака
        price_changes = np.diff(price_data, axis=1)  # (batch_size, seq_len-1, num_features)

        # Создаем спайковый тензор
        spike_tensor = np.zeros((batch_size, 3, seq_len, self.num_time_steps))

        for batch_idx in range(batch_size):
            for feature_idx in range(num_features):
                feature_changes = price_changes[batch_idx, :, feature_idx]

                if len(feature_changes) == 0:
                    continue

                # Нормализация изменений для каждого признака
                mean_change = np.mean(feature_changes)
                std_change = np.std(feature_changes)

                if std_change < 1e-8:
                    normalized_changes = np.zeros_like(feature_changes)
                else:
                    normalized_changes = (feature_changes - mean_change) / std_change

                # Генерация спайков на основе порогов
                for step_idx, change in enumerate(normalized_changes):
                    if step_idx >= seq_len - 1:  # Защита от выхода за границы
                        continue

                    if change > self.threshold_std:
                        # Спайк вверх
                        spike_indices = np.linspace(0, self.num_time_steps - 1, 3, dtype=int)
                        spike_tensor[batch_idx, 0, step_idx, spike_indices] = 1
                    elif change < -self.threshold_std:
                        # Спайк вниз
                        spike_indices = np.linspace(0, self.num_time_steps - 1, 3, dtype=int)
                        spike_tensor[batch_idx, 1, step_idx, spike_indices] = 1
                    else:
                        # Базовый уровень активности
                        spike_indices = np.random.choice(self.num_time_steps, 1)
                        spike_tensor[batch_idx, 2, step_idx, spike_indices] = 1

        return torch.FloatTensor(spike_tensor)


# =============================================================================
# 3. АРХИТЕКТУРА СПАЙКОВОЙ RESNEXT (БЕЗ СОХРАНЕНИЯ СОСТОЯНИЯ)
# =============================================================================

class MultiTimeframeSpikingResNeXt(nn.Module):
    """Спайковая ResNeXt для мультитаймфреймового анализа EURUSD"""

    def __init__(self, num_timeframes=4, num_classes=3, num_steps=50,
                 cardinality=16, beta=0.9):
        super().__init__()

        self.num_steps = num_steps
        self.num_timeframes = num_timeframes

        # ОДНА общая входная свертка для всех таймфреймов
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta, learn_beta=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNeXt блоки для обработки features
        self.resnext_blocks = nn.Sequential(
            SparseSpikingResNeXtBlock(64, 128, stride=2, cardinality=cardinality),
            SparseSpikingResNeXtBlock(128, 256, stride=2, cardinality=cardinality),
            SparseSpikingResNeXtBlock(256, 512, stride=1, cardinality=cardinality),
        )

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(512 * num_timeframes, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, num_timeframes, 3, height, width, num_steps)
        batch_size = x.shape[0]
        num_tf = x.shape[1]
        height = x.shape[3]
        width = x.shape[4]
        num_steps = x.shape[5]

        # Инициализация мембранных потенциалов (ВРЕМЕННЫЕ для этого вызова)
        mem1 = self.lif1.init_leaky()

        # Обрабатываем каждый таймфрейм и временной шаг
        timeframe_features = []

        for tf_idx in range(num_tf):
            tf_step_features = []

            for step in range(num_steps):
                x_tf_step = x[:, tf_idx, :, :, :, step]  # (batch, 3, height, width)

                # Применяем входную свертку
                x_conv = self.conv1(x_tf_step)
                x_conv = self.bn1(x_conv)
                x_conv, mem1 = self.lif1(x_conv, mem1)  # используем временную память
                x_conv = self.pool1(x_conv)

                # Применяем ResNeXt блоки с временной памятью
                x_features = self._forward_resnext_blocks(x_conv)

                # Global average pooling для получения feature vector
                x_pooled = F.adaptive_avg_pool2d(x_features, (1, 1))
                x_pooled = x_pooled.view(batch_size, -1)  # (batch, 512)

                tf_step_features.append(x_pooled)

            # Усредняем по временным шагам для данного таймфрейма
            tf_all_steps = torch.stack(tf_step_features, dim=0)  # (num_steps, batch, 512)
            tf_avg = tf_all_steps.mean(dim=0)  # (batch, 512)
            timeframe_features.append(tf_avg)

        # Объединяем features от всех таймфреймов
        combined_features = torch.cat(timeframe_features, dim=1)  # (batch, num_tf * 512)

        # Классификация
        output = self.classifier(combined_features)

        return output

    def _forward_resnext_blocks(self, x):
        """Прямой проход через ResNeXt блоки с временной памятью"""
        for block in self.resnext_blocks:
            # Для каждого блока создаем временные мембранные потенциалы
            if isinstance(block, SparseSpikingResNeXtBlock):
                x = block.forward_with_temp_mem(x)
            else:
                x = block(x)
        return x

    def reset_mem(self):
        """Теперь эта функция не нужна, но оставляем для совместимости"""
        pass


class SparseSpikingResNeXtBlock(nn.Module):
    """Спайковый блок ResNeXt с групповыми свертками"""

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32,
                 width_factor=4, beta=0.9):
        super().__init__()

        self.beta = beta
        intermediate_channels = cardinality * width_factor

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels,
                               kernel_size=3, stride=stride, padding=1,
                               groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        self.conv3 = nn.Conv2d(intermediate_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.lif1 = snn.Leaky(beta=beta, learn_beta=True)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=True)
        self.lif3 = snn.Leaky(beta=beta, learn_beta=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.lif_shortcut = snn.Leaky(beta=beta, learn_beta=True)
            self.use_shortcut_lif = True
        else:
            self.use_shortcut_lif = False

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Используем временную память по умолчанию
        return self.forward_with_temp_mem(x)

    def forward_with_temp_mem(self, x):
        """Прямой проход с временными мембранными потенциалами"""
        # Инициализация временной памяти
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        if self.use_shortcut_lif:
            mem_shortcut = self.lif_shortcut.init_leaky()

        # Основной путь
        out = self.conv1(x)
        out = self.bn1(out)
        out, mem1 = self.lif1(out, mem1)

        out = self.conv2(out)
        out = self.bn2(out)
        out, mem2 = self.lif2(out, mem2)

        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut путь
        residual = self.shortcut(x)
        if self.use_shortcut_lif:
            residual, mem_shortcut = self.lif_shortcut(residual, mem_shortcut)

        # Сложение и финальный спайк
        out += residual
        out, mem3 = self.lif3(out, mem3)

        return out

    def reset_mem(self):
        """Теперь не нужен"""
        pass


# =============================================================================
# 4. ИСПРАВЛЕННЫЙ ДАТАСЕТ
# =============================================================================

class EURUSDDataset(Dataset):
    """Датасет для мультитаймфреймовых данных EURUSD"""

    def __init__(self, data_dict, lookback_window=100, num_steps=50, target_timeframe='M5'):
        self.data_dict = data_dict
        self.lookback_window = lookback_window
        self.num_steps = num_steps
        self.target_timeframe = target_timeframe
        self.encoder = FinancialSpikeEncoder(num_time_steps=num_steps)

        self.expected_timeframes = ['M5', 'M15', 'H1', 'H4']

        self._validate_and_preprocess_data()
        self._create_timeline()

    def _validate_and_preprocess_data(self):
        self.processed_data = {}
        self.available_timeframes = []

        print("🔍 Проверяем доступные таймфреймы...")

        for tf_name in self.expected_timeframes:
            if tf_name in self.data_dict and self.data_dict[tf_name] is not None:
                df = self.data_dict[tf_name]

                if len(df) > self.lookback_window:
                    self.available_timeframes.append(tf_name)
                    print(f"   ✅ {tf_name}: {len(df)} баров")

                    features = self._preprocess_dataframe(df)
                    self.processed_data[tf_name] = features
                else:
                    print(f"   ⚠️ {tf_name}: недостаточно данных ({len(df)} баров)")
            else:
                print(f"   ❌ {tf_name}: данные отсутствуют")

        if not self.available_timeframes:
            raise ValueError("❌ Нет доступных таймфреймов с достаточным количеством данных")

        if self.target_timeframe not in self.processed_data:
            if 'M5' in self.processed_data:
                self.target_timeframe = 'M5'
            else:
                self.target_timeframe = self.available_timeframes[0]
            print(f"⚠️ Целевой таймфрейм изменен на {self.target_timeframe}")

        print(f"🎯 Целевой таймфрейм: {self.target_timeframe}")
        print(f"📊 Доступные таймфреймы: {self.available_timeframes}")

    def _preprocess_dataframe(self, df):
        required_columns = ['open', 'high', 'low', 'close', 'tick_volume']

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            available_cols = list(df.columns)
            raise ValueError(f"Отсутствуют колонки: {missing_columns}. Доступны: {available_cols}")

        features = df[required_columns].values

        # Логарифмирование цен для стабильности
        price_columns = ['open', 'high', 'low', 'close']
        price_indices = [required_columns.index(col) for col in price_columns]

        for idx in price_indices:
            features[:, idx] = np.log(features[:, idx])

        # Нормализация объема
        volume_idx = required_columns.index('tick_volume')
        features[:, volume_idx] = np.log1p(features[:, volume_idx])

        return features

    def _create_timeline(self):
        target_data = self.processed_data[self.target_timeframe]

        self.valid_indices = list(range(
            self.lookback_window,
            len(target_data) - 1
        ))

        print(f"📈 Доступно samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        multi_tf_data = []

        timeframe_ratios = {
            'M5': 1,
            'M15': 3,
            'H1': 12,
            'H4': 48
        }

        for tf_name in self.available_timeframes:
            data = self.processed_data[tf_name]
            ratio = timeframe_ratios.get(tf_name, 1)

            window_start = max(0, (real_idx - self.lookback_window) // ratio)
            window_end = real_idx // ratio

            if window_end >= len(data):
                window_end = len(data) - 1
                window_start = max(0, window_end - (self.lookback_window // ratio))

            window_data = data[window_start:window_end]

            if len(window_data) < self.lookback_window:
                padding_needed = self.lookback_window - len(window_data)
                padding = np.tile(window_data[0:1], (padding_needed, 1))
                window_data = np.vstack([padding, window_data])

            # Кодируем в спайки
            window_data_reshaped = window_data.reshape(1, self.lookback_window, 5)
            spikes = self.encoder.price_change_encoding(window_data_reshaped)

            # spikes shape: (1, 3, lookback_window, num_steps)
            # Преобразуем к (3, lookback_window, num_steps)
            spikes = spikes.squeeze(0)

            # ИСПРАВЛЕНО: добавляем измерение высоты (делаем lookback_window как height)
            # Преобразуем (3, lookback_window, num_steps) -> (3, lookback_window, 1, num_steps)
            spikes = spikes.unsqueeze(2)  # добавляем width=1

            multi_tf_data.append(spikes)

        # Создаем общий тензор: (num_timeframes, 3, height, width, num_steps)
        # где height = lookback_window, width = 1
        multi_tf_tensor = torch.stack(multi_tf_data)

        # Целевая переменная
        target_data = self.processed_data[self.target_timeframe]
        future_price = target_data[real_idx + 1, 3]
        current_price = target_data[real_idx, 3]

        price_ratio = np.exp(future_price - current_price)

        threshold = 1.0005 if self.target_timeframe == 'M5' else 1.001

        if price_ratio > threshold:
            target = 2  # UP
        elif price_ratio < (2 - threshold):
            target = 0  # DOWN
        else:
            target = 1  # SIDEWAYS

        return multi_tf_tensor, torch.tensor(target, dtype=torch.long)

def create_mock_eurusd_data():
    """Создание тестовых данных для EURUSD"""
    print("🔄 Создаем тестовые данные EURUSD...")

    base_dates = pd.date_range(start='2024-01-01', end='2025-01-01', freq='5min')[:5000]

    mock_data = {}

    timeframe_configs = [
        ('M5', 1),
        ('M15', 3),
        ('H1', 12),
        ('H4', 48)
    ]

    # Создаем базовые данные для M5 с реалистичным поведением EURUSD
    np.random.seed(42)
    base_prices = []
    price = 1.1000  # Начальная цена EURUSD

    for i in range(len(base_dates)):
        # Более реалистичная модель движения EURUSD
        trend = 0.00001  # Слабый восходящий тренд
        volatility = 0.0008  # Волатильность EURUSD
        change = np.random.normal(trend, volatility)
        price = max(0.9, min(1.3, price * (1 + change)))  # Ограничиваем разумные значения
        base_prices.append(price)

    for tf_name, multiplier in timeframe_configs:
        data = []

        for i in range(0, len(base_prices), multiplier):
            if i >= len(base_prices):
                break

            base_idx = i
            open_price = base_prices[base_idx]

            period_high = max(base_prices[base_idx:min(base_idx + multiplier, len(base_prices))])
            period_low = min(base_prices[base_idx:min(base_idx + multiplier, len(base_prices))])

            close_idx = min(base_idx + multiplier - 1, len(base_prices) - 1)
            close_price = base_prices[close_idx]

            volume = np.random.randint(100 * multiplier, 1000 * multiplier)

            data.append({
                'open': open_price,
                'high': period_high,
                'low': period_low,
                'close': close_price,
                'tick_volume': volume
            })

        df = pd.DataFrame(data)

        if tf_name == 'M5':
            df.index = base_dates[:len(df)]
        else:
            df.index = base_dates[::multiplier][:len(df)]

        mock_data[tf_name] = df
        print(f"✅ {tf_name}: создано {len(df)} тестовых баров")

    return mock_data


def verify_onnx_model(onnx_path):
    """Проверка валидности ONNX модели"""
    try:
        # Загружаем модель
        onnx_model = onnx.load(onnx_path)

        # Проверяем валидность
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX модель валидна")

        # Проверяем входы/выходы
        print("📊 Граф модели:")
        for input in onnx_model.graph.input:
            print(f"   Вход: {input.name}, форма: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

        for output in onnx_model.graph.output:
            print(f"   Выход: {output.name}, форма: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")

        # Тестируем inference
        session = ort.InferenceSession(onnx_path)
        print("✅ ONNX Runtime сессия создана успешно")

        return True

    except Exception as e:
        print(f"❌ Ошибка проверки ONNX: {e}")
        return False

def export_to_onnx(model, model_path, input_shape, device='cuda'):
    """
    Экспорт модели в ONNX формат на CUDA
    """
    try:
        # Переводим модель в режим оценки и на CUDA
        model.eval()
        model = model.to(device)  # Переносим модель на CUDA

        # Создаем фиктивный вход на CUDA
        dummy_input = torch.randn(input_shape, device=device)  # Создаем на CUDA

        # Экспорт в ONNX
        torch.onnx.export(
            model,
            dummy_input,
            model_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=True
        )

        print(f"✅ Модель успешно экспортирована в: {model_path}")

        # Проверяем размер файла
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📦 Размер ONNX файла: {file_size:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ Ошибка экспорта в ONNX: {e}")
        return False


if __name__ == "__main__":
    # Проверяем доступность CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Используется устройство: {device}")

    if not torch.cuda.is_available():
        print("❌ CUDA недоступно! Экспорт будет на CPU")
        device = 'cpu'

    try:
        # 1. Определяем форму входа
        print("📥 Создаем тестовые данные для определения формы...")
        data_dict = create_mock_eurusd_data()
        dataset = EURUSDDataset(
            data_dict=data_dict,
            lookback_window=80,
            num_steps=50,
            target_timeframe='M5'
        )
        sample_input, _ = dataset[0]
        input_shape = (1, sample_input.shape[0], sample_input.shape[1],
                       sample_input.shape[2], sample_input.shape[3], sample_input.shape[4])
        print(f"📐 Форма входа: {input_shape}")

        # 2. Проверяем наличие файла с весами
        model_path = "saved_models/epoch_8_train_34.17_val_35.02.pth"
        if not os.path.exists(model_path):
            print(f"❌ Файл модели не найден: {model_path}")
            exit()

        # 3. Создаем архитектуру модели
        print("🧠 Создаем архитектуру модели...")
        model = MultiTimeframeSpikingResNeXt(
            num_timeframes=len(dataset.available_timeframes),
            num_classes=3,
            num_steps=50
        )

        # 4. Загружаем веса с указанием устройства
        print("📥 Загружаем веса модели...")
        state_dict = torch.load(model_path, map_location=device)  # Загружаем на нужное устройство
        model.load_state_dict(state_dict)

        # 5. Переносим модель на нужное устройство
        model = model.to(device)
        print(f"✅ Модель загружена и перенесена на {device}")

        # 6. Переводим модель в режим оценки
        model.eval()
        print("🔍 Модель переведена в режим оценки")

        # 7. Экспортируем в ONNX
        print("🔄 Экспортируем модель в ONNX...")
        success = export_to_onnx(model, 'eurusd_spiking_resnext_eph8_35.02.onnx', input_shape, device)

        if success:
            print("🎉 ONNX экспорт завершен успешно!")

            # Проверяем ONNX модель
            verify_onnx_model('eurusd_spiking_resnext_eph8_35.02.onnx')
        else:
            print("❌ ONNX экспорт не удался")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()