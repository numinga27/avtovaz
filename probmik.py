import cv2
import numpy as np

def load_template_images():
    # Загрузите изображения шаблонов для правой и левой деталей
    left_template = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
    right_template = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE)
    return left_template, right_template

def preprocess_image(image_path):
    # Загрузка и предварительная обработка изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Применение размытия
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  # Бинаризация
    return image

def compare_images(input_image, left_template, right_template):
    # Сравнение входного изображения с шаблонами
    left_result = cv2.matchTemplate(input_image, left_template, cv2.TM_CCOEFF_NORMED)
    right_result = cv2.matchTemplate(input_image, right_template, cv2.TM_CCOEFF_NORMED)

    # Получение максимальных значений совпадения
    left_max_val = np.max(left_result)
    right_max_val = np.max(right_result)

    return left_max_val, right_max_val

def main(image_path):
    left_template, right_template = load_template_images()
    input_image = preprocess_image(image_path)

    left_score, right_score = compare_images(input_image, left_template, right_template)

    if left_score > right_score:
        print("Левая деталь")
    else:
        print("Правая деталь")

if __name__ == "__main__":
    main('input_image.jpg')  # Укажите путь к вашему изображению
