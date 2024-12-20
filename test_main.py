import cv2
import numpy as np


def process_image(image_path, area_ranges):
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: изображение не загружено. Проверьте путь к файлу.")
        return

    # Преобразование в градации серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение пороговой обработки для инверсии цветов
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)

    # Нахождение контуров
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []

    for cnt in contours:
        # Вычисление площади
        area = cv2.contourArea(cnt)

        # Вычисление центра масс
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            center_of_mass = (int(M["m10"] / M["m00"]),
                              int(M["m01"] / M["m00"]))
            # Вычисление момента инерции
            inertia = (M["m20"] * M["m00"] - (M["m10"] * M["m10"]) / M["m00"]) + \
                (M["m02"] * M["m00"] - (M["m01"] * M["m01"]) / M["m00"])
            
        else:
            center_of_mass = (0, 0)
            inertia = 0  # Устанавливаем момент инерции в 0, если площадь равна 0

        # Проверка на соответствие заданным диапазонам
        if area_ranges['right']['min'] <= area <= area_ranges['right']['max']:
            results.append(
                f"Правая деталь найдена! Центр масс: {center_of_mass}, Площадь: {area}, Момент инерции: {inertia}")
        elif area_ranges['left']['min'] <= area <= area_ranges['left']['max']:
            results.append(
                f"Левая деталь найдена! Центр масс: {center_of_mass}, Площадь: {area}, Момент инерции: {inertia}")

    # Вывод результатов
    if results:
        for result in results:
            print(result)
    else:
        print("Детали не найдены в заданных диапазонах.")


# Укажите путь к вашему изображению и диапазоны площадей
image_path = '1.png'
area_ranges = {
    'right': {'min': 1700.0, 'max': 1851.0},  # Диапазон для правой детали
    'left': {'min': 1000.0, 'max': 1155.0}   # Диапазон для левой детали
}


process_image(image_path, area_ranges)
