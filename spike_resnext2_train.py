import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')


# =============================================================================
# 1. ФУНКЦИИ ДЛЯ РАБОТЫ С META TRADER 5
# =============================================================================

def initialize_mt5():
    """Инициализация подключения к MetaTrader5"""
    if not mt5.initialize():
        print("❌ Не удалось инициализировать MT5")
        return False
    print("✅ MT5 успешно инициализирован")
    return True


def download_mt5_data(symbol="EURUSDrfd", bars_count=20000):
    """
    Загрузка данных из MetaTrader5 для EURUSD
    """
    timeframes = {
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4
    }

    data_dict = {}

    for tf_name, tf_enum in timeframes.items():
        try:
            print(f"📥 Загружаем данные для {symbol} {tf_name}...")

            rates = mt5.copy_rates_from_pos(symbol, tf_enum, 0, bars_count)

            if rates is None:
                print(f"❌ Не удалось загрузить данные для {tf_name}")
                continue

            df = pd.DataFrame(rates)
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('datetime', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'tick_volume']]
            data_dict[tf_name] = df

            print(f"✅ {tf_name}: загружено {len(df)} баров")

        except Exception as e:
            print(f"❌ Ошибка при загрузке {tf_name}: {e}")

    return data_dict


def create_mock_eurusd_data():
    """Создание тестовых данных для EURUSD"""
    print("🔄 Создаем тестовые данные EURUSD...")

    base_dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='5min')[:5000]

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


# =============================================================================
# 2. ИСПРАВЛЕННЫЙ КОДИРОВЩИК ДАННЫХ В СПАЙКИ
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

        # Используйте реалистичные пороги (0.1-0.5%)
        threshold = 0.00007 if self.target_timeframe == 'M5' else 0.00005

        if price_ratio > 1 + threshold:
            target = 2  # BUY
        elif price_ratio < 1 - threshold:
            target = 0  # SELL
        else:
            target = 1  # HOLD

        return multi_tf_tensor, torch.tensor(target, dtype=torch.long)

# =============================================================================
# 5+ Проверьте реальное распределение предсказаний
# =============================================================================
def analyze_dataset(dataloader):
    targets_count = {0: 0, 1: 0, 2: 0}  # SELL, HOLD, BUY

    for _, targets in dataloader:
        # targets - это тензор с размерностью [batch_size]
        for target in targets:
            cls = target.item()  # теперь это скаляр
            targets_count[cls] = targets_count.get(cls, 0) + 1

    total_samples = sum(targets_count.values())
    print("=== РАСПРЕДЕЛЕНИЕ TARGETS В ДАТАСЕТЕ ===")
    for cls, count in targets_count.items():
        class_name = ["SELL", "HOLD", "BUY"][cls]
        percentage = count / total_samples * 100
        print(f"{class_name}: {count} samples ({percentage:.1f}%)")

    return targets_count


def analyze_model_predictions(model, dataloader, device):
    model.eval()
    model.to(device)  # Убедимся, что модель на правильном устройстве

    predictions_count = {0: 0, 1: 0, 2: 0}
    correct_predictions = {0: 0, 1: 0, 2: 0}
    total_predictions = 0

    with torch.no_grad():
        for data, targets in dataloader:
            # Переносим данные на то же устройство, что и модель
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            predictions = outputs.argmax(dim=1)  # [batch_size]

            # Переносим обратно на CPU для подсчета
            predictions_cpu = predictions.cpu()
            targets_cpu = targets.cpu()

            for pred, target in zip(predictions_cpu, targets_cpu):
                pred_cls = pred.item()
                target_cls = target.item()

                predictions_count[pred_cls] += 1
                if pred_cls == target_cls:
                    correct_predictions[pred_cls] += 1
                total_predictions += 1

    print("\n=== РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ МОДЕЛИ ===")
    for cls in range(3):
        class_name = ["SELL", "HOLD", "BUY"][cls]
        pred_count = predictions_count[cls]
        correct_count = correct_predictions[cls]
        pred_percentage = pred_count / total_predictions * 100 if total_predictions > 0 else 0

        if pred_count > 0:
            accuracy = correct_count / pred_count * 100
        else:
            accuracy = 0

        print(f"{class_name}: {pred_count} preds ({pred_percentage:.1f}%), Accuracy: {accuracy:.1f}%")

    return predictions_count


def plot_confusion_matrix(model, dataloader, device):
    model.eval()
    model.to(device)

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            predictions = outputs.argmax(dim=1)

            # Переносим на CPU для анализа
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    cm = confusion_matrix(all_targets, all_predictions)
    classes = ["SELL", "HOLD", "BUY"]

    print("\n=== МАТРИЦА ОШИБОК ===")
    print("Rows = True, Columns = Predicted")
    print("     SELL HOLD BUY")
    for i, class_name in enumerate(classes):
        row = f"{class_name}: "
        for j in range(3):
            row += f"{cm[i][j]:4d} "
        print(row)

    # Вычисляем precision для каждого класса
    print("\n=== ДЕТАЛЬНАЯ СТАТИСТИКА ===")
    for i, class_name in enumerate(classes):
        precision = cm[i][i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
        recall = cm[i][i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        print(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}")

    return cm


def analyze_price_changes(data_dict, target_timeframe='M5'):
    """Анализ реальных изменений цен"""
    if target_timeframe not in data_dict:
        print(f"❌ Таймфрейм {target_timeframe} не найден в данных")
        return

    df = data_dict[target_timeframe]

    # Преобразуем в numpy array если нужно
    if hasattr(df, 'values'):
        data = df.values
    else:
        data = df

    price_changes = []

    for i in range(1, len(data)):
        # Правильное получение цен (индекс 3 для close price в массиве)
        current_price = data[i - 1][3] if len(data[i - 1]) > 3 else data[i - 1][0]  # close price
        future_price = data[i][3] if len(data[i]) > 3 else data[i][0]  # next close price

        change = (future_price - current_price) / current_price
        price_changes.append(change)

    price_changes = np.array(price_changes)

    print("=== АНАЛИЗ ИЗМЕНЕНИЙ ЦЕН ===")
    print(f"Таймфрейм: {target_timeframe}")
    print(f"Образцов: {len(price_changes)}")
    print(f"Среднее изменение: {np.mean(price_changes) * 100:.3f}%")
    print(f"Стандартное отклонение: {np.std(price_changes) * 100:.3f}%")
    print(f"Максимальный рост: {np.max(price_changes) * 100:.3f}%")
    print(f"Максимальное падение: {np.min(price_changes) * 100:.3f}%")

    # Тестируем разные пороги
    thresholds = [0.00005, 0.00007, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]  # 0.01% до 0.5%

    print(f"\n{'Порог':>8} {'BUY':>8} {'SELL':>8} {'HOLD':>8} {'B+S':>8}")
    print("-" * 50)

    for threshold in thresholds:
        buy_count = np.sum(price_changes > threshold)
        sell_count = np.sum(price_changes < -threshold)
        hold_count = len(price_changes) - buy_count - sell_count

        print(f"{threshold * 100:7.3f}% {buy_count:7d} {sell_count:7d} {hold_count:7d} {buy_count + sell_count:7d}")

    # Автоматический подбор порога для баланса
    print(f"\n=== АВТОМАТИЧЕСКИЙ ПОДБОР ПОРОГА ===")
    target_ratio = 0.3  # Целевая доля сигналов (30% BUY + 30% SELL)

    best_threshold = 0.001  # по умолчанию 0.1%
    best_balance = float('inf')

    for threshold in np.linspace(0.00001, 0.01, 50):  # от 0.001% до 1%
        buy_count = np.sum(price_changes > threshold)
        sell_count = np.sum(price_changes < -threshold)
        signal_ratio = (buy_count + sell_count) / len(price_changes)

        # Ищем порог, дающий близкое к целевому соотношение
        balance = abs(signal_ratio - target_ratio * 2) * 2
        # сигнала(BUY + SELL)

        if balance < best_balance:
            best_balance = balance
            best_threshold = threshold

    buy_count = np.sum(price_changes > best_threshold)
    sell_count = np.sum(price_changes < -best_threshold)
    hold_count = len(price_changes) - buy_count - sell_count

    print(f"Рекомендуемый порог: {best_threshold * 100:.3f}%")
    print(f"Распределение при этом пороге:")
    print(f"  BUY: {buy_count} ({buy_count / len(price_changes) * 100:.1f}%)")
    print(f"  SELL: {sell_count} ({sell_count / len(price_changes) * 100:.1f}%)")
    print(f"  HOLD: {hold_count} ({hold_count / len(price_changes) * 100:.1f}%)")

    return best_threshold

# =============================================================================
# 5. ОБУЧЕНИЕ МОДЕЛИ
# =============================================================================

def train_eurusd_model():
    """Функция обучения модели для EURUSD"""

    print("🚀 Запуск обучения спайковой ResNeXt для EURUSD")

    # ПРОВЕРКА И ВЫБОР УСТРОЙСТВА
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Используется устройство: {device}")

    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        print(f"🎯 CUDA версия: {torch.version.cuda}")
        print(f"🎯 Доступно GPU памяти: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("⚠️  CUDA недоступно, используется CPU (обучение будет медленным)")

    mt5_available = initialize_mt5()

    if mt5_available:
        try:
            data_dict = download_mt5_data(symbol="EURUSDrfd", bars_count=20000)
        except Exception as e:
            print(f"❌ Ошибка загрузки из MT5: {e}")
            data_dict = create_mock_eurusd_data()
    else:
        data_dict = create_mock_eurusd_data()

    try:
        # Создаем полный датасет
        full_dataset = EURUSDDataset(
            data_dict=data_dict,
            lookback_window=80,
            num_steps=50,
            target_timeframe='M5'
        )

        print(f"\n✅ Датасет создан успешно!")
        print(f"   Доступные таймфреймы: {full_dataset.available_timeframes}")
        print(f"   Целевой таймфрейм: {full_dataset.target_timeframe}")
        print(f"   Всего samples: {len(full_dataset)}")

        # РАЗДЕЛЕНИЕ НА TRAIN/VALIDATION
        train_size = int(0.8 * len(full_dataset))  # 80% для обучения
        val_size = len(full_dataset) - train_size  # 20% для валидации

        # Используем random_split для сохранения распределения классов
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # для воспроизводимости
        )

        print(f"📊 Разделение данных:")
        print(f"   Train samples: {len(train_dataset)} ({len(train_dataset) / len(full_dataset) * 100:.1f}%)")
        print(f"   Validation samples: {len(val_dataset)} ({len(val_dataset) / len(full_dataset) * 100:.1f}%)")

        # СОЗДАЕМ DATALOADER ДЛЯ TRAIN И VALIDATION
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # shuffle=False для валидации

        # # ПРОВЕРКА РАСПРЕДЕЛЕНИЯ TARGETS
        # print("\n=== РАСПРЕДЕЛЕНИЕ TARGETS ===")
        # print("Train dataset:")
        # analyze_dataset(train_dataloader)
        # print("\nValidation dataset:")
        # analyze_dataset(val_dataloader)
        #
        # # АНАЛИЗ ИЗМЕНЕНИЙ ЦЕН
        # recommended_threshold = analyze_price_changes(data_dict, 'M5')
        # print(f"Recommended threshold: {recommended_threshold}")

        # СОЗДАНИЕ МОДЕЛИ
        model = MultiTimeframeSpikingResNeXt(
            num_timeframes=len(full_dataset.available_timeframes),
            num_classes=3,
            num_steps=50,
            cardinality=16,
            beta=0.9
        )

        # ЗАГРУЗКА ВЕСОВ (ЕСЛИ ЕСТЬ)
        print("📥 Загружаем веса модели...")
        model_path = "saved_models/epoch_7_train_33.92_val_34.91.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Веса модели загружены")
        else:
            print("⚠️  Файл весов не найден, начинаем со случайной инициализации")

        # ПЕРЕНЕСТИ МОДЕЛЬ НА УСТРОЙСТВО
        model = model.to(device)
        print(f"🧠 Модель создана для {len(full_dataset.available_timeframes)} таймфреймов и перенесена на {device}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=2,  # Более агрессивный - ждет только 2 эпохи
            factor=0.3,  # Сильнее уменьшает LR (в 3 раза)
            min_lr=1e-7  # Абсолютный минимум
        )

        best_accuracy = 34.91
        val_accuracy = 34.91  # Вставить здесь показатель той модели, с которой стартуем
        models_dir = "saved_models"
        os.makedirs(models_dir, exist_ok=True)

        print("\n🎯 Начинаем обучение...")

        for epoch in range(8, 15):
            # === ФАЗА ОБУЧЕНИЯ ===
            model.train()
            train_total_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, targets) in enumerate(train_dataloader):
                data = data.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scheduler.step(val_accuracy)

                train_total_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

                current_lr = optimizer.param_groups[0]['lr']

                if batch_idx % 50 == 0:
                    current_accuracy = 100. * predicted.eq(targets).sum().item() / targets.size(0)
                    print(f'{datetime.now()}, Epoch: {epoch}, Train Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                          f'Batch Accuracy: {current_accuracy:.2f}%, LR: {current_lr:.2e}')

            train_accuracy = 100. * train_correct / train_total
            train_avg_loss = train_total_loss / len(train_dataloader)

            # === ФАЗА ВАЛИДАЦИИ ===
            model.eval()
            val_total_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, targets in val_dataloader:
                    data = data.to(device)
                    targets = targets.to(device)

                    outputs = model(data)
                    loss = criterion(outputs, targets)

                    val_total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_accuracy = 100. * val_correct / val_total
            val_avg_loss = val_total_loss / len(val_dataloader)

            # === ВЫВОД РЕЗУЛЬТАТОВ ===
            print(f'\n📊 Epoch: {epoch}')
            print(f'   Train - Loss: {train_avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
            print(f'   Val   - Loss: {val_avg_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
            print(f'   Overfitting: {train_accuracy - val_accuracy:+.2f}%')

            # === СОХРАНЕНИЕ МОДЕЛЕЙ ===
            # Сохраняем каждую эпоху
            torch.save(model.state_dict(),
                       f'{models_dir}/epoch_{epoch}_train_{train_accuracy:.2f}_val_{val_accuracy:.2f}.pth')
            print(f"💾 Модель сохранена: epoch_{epoch}")

            # Сохраняем лучшую модель по валидационной accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), f'{models_dir}/best_model_val_acc_{val_accuracy:.2f}.pth')
                print(f"🏆 Новая лучшая модель сохранена: val_acc={val_accuracy:.2f}%")

            # Сохраняем последнюю модель
            torch.save(model.state_dict(), f'{models_dir}/latest_model.pth')

            # # === АНАЛИЗ ПРЕДСКАЗАНИЙ НА VALIDATION ===
            # if epoch % 2 == 0:  # Каждые 2 эпохи
            #     print(f"\n🔍 Анализ предсказаний на валидации (эпоха {epoch}):")
            #     analyze_model_predictions(model, val_dataloader, device)

        # Сохраняем финальную модель
        torch.save(model.state_dict(), f'{models_dir}/final_model_val_acc_{val_accuracy:.2f}.pth')
        print(f"✅ Финальная модель сохранена: val_acc={val_accuracy:.2f}%")

        print("\n✅ Обучение завершено!")

        # ФИНАЛЬНЫЙ АНАЛИЗ
        print("\n=== ФИНАЛЬНЫЙ АНАЛИЗ ===")
        # print("Train dataset predictions:")
        # analyze_model_predictions(model, train_dataloader, device)
        print("\nValidation dataset predictions:")
        analyze_model_predictions(model, val_dataloader, device)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 6. ЗАПУСК ПРОГРАММЫ
# =============================================================================

if __name__ == "__main__":
    train_eurusd_model()