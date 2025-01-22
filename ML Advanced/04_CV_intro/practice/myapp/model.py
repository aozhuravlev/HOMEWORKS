import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.ndimage import binary_dilation


# Определение класса SimpleCNN (если он не импортируется из другого модуля)import torch
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Свёрточные слои + BatchNorm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Полносвязные слои + BatchNorm + Dropout
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(
            -1, 1, 28, 28
        )  # Убедимся, что входные данные имеют нужную размерность

        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Разворачивание перед FC-слоями

        # x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        # x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = F.leaky_relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)

        return x


# Класс для загрузки модели и выполнения предсказаний
class Model:
    def __init__(self):
        # Указываем путь к файлу модели
        model_path = os.path.join("model", "EMNIST_CNN.pth")

        # Загружаем модель и метаданные
        checkpoint = torch.load(model_path, map_location="cpu")
        self.num_classes = checkpoint["metadata"]["num_classes"]

        # Инициализируем модель и загружаем параметры
        self.model = SimpleCNN(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()  # Переводим модель в режим оценки

        # Загружаем метаданные (если нужно)
        self.metadata = checkpoint["metadata"]

    def predict(self, x):
        """
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание
        """
        # Применяем дилатацию
        x = binary_dilation(x).astype(np.float32)

        # Нормализация
        mean = 0.1307
        std = 0.3081
        x = (x - mean) / std

        # Преобразуем входное изображение в тензор
        x = torch.from_numpy(x).float()
        x = x.unsqueeze(0).unsqueeze(0)

        # Выполняем предсказание
        with torch.no_grad():
            outputs = self.model(x)
            _, predicted = torch.max(outputs, 1)

        # Преобразуем предсказание в символ (если нужно)
        pred = self._map_class_to_symbol(predicted.item())
        return pred

    def _map_class_to_symbol(self, class_index):
        """
        Преобразует индекс класса в символ (если нужно).
        Здесь можно добавить логику для преобразования индекса в символ.
        """
        symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
        return symbols[class_index]
