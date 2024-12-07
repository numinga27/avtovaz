import unittest
import cv2
import numpy as np
# Предполагаем, что функция check_black_cap находится в main.py
from main import check_black_cap


class TestBlackCapDetection(unittest.TestCase):

    def test_black_cap_detected(self):
        # Создаем изображение с черным колпачком
        black_cap_image = np.zeros(
            (100, 100, 3), dtype=np.uint8)  # Черное изображение
        # Добавляем белый круг для теста
        cv2.circle(black_cap_image, (50, 50), 20, (255, 255, 255), -1)
        result = check_black_cap(black_cap_image)
        self.assertTrue(result)  # Ожидаем, что черный колпачок будет найден

    def test_black_cap_not_detected(self):
        # Создаем изображение без черного колпачка
        no_cap_image = np.ones(
            (100, 100, 3), dtype=np.uint8) * 255  # Белое изображение
        result = check_black_cap(no_cap_image)
        # Ожидаем, что черный колпачок не будет найден
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
