import cv2
import numpy as np

# Загрузка изображения
image_path = '2.png'  # Укажите путь к вашему изображению
image = cv2.imread(image_path)

if image is None:
    print("Ошибка: изображение не загружено. Проверьте путь к файлу.")
    exit()

# Преобразование в черно-белый формат
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Создание маски для черных объектов (все, что не черное, становится белым)
_, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# Нахождение контуров черных фигур
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Список для хранения информации о черных фигурах
black_figures_info = []

# Эталонные значения (пример)
reference_values = {
    'cap': {'area': 5000, 'inertia': 1000},  # Примерные значения для колпачка
    # Примерные значения для другой детали
    'other_part': {'area': 3000, 'inertia': 800}
}

# Обработка контуров черных фигур
for contour in contours:
    area = cv2.contourArea(contour)

    if area < 100:  # Фильтрация по площади (можно настроить)
        continue

    # Вычисление центра масс
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Сохранение информации о черной фигуре
    black_figures_info.append({
        'area': area,
        'center': (cX, cY),
        'contour': contour
    })

# Обработка серых фигур (например, для нахождения момента инерции)
gray_figures_info = []

for contour in contours:
    # Создаем маску для серой фигуры
    mask_gray = np.zeros_like(gray)
    cv2.drawContours(mask_gray, [contour], -1, (255), thickness=cv2.FILLED)

    # Вычисляем момент инерции
    M_gray = cv2.moments(mask_gray)
    inertia = (M_gray["mu20"] + M_gray["mu02"]) / \
        M_gray["m00"] if M_gray["m00"] != 0 else 0

    gray_figures_info.append({
        'inertia': inertia,
        'contour': contour
    })

# Сравнение с эталонными значениями
for figure in gray_figures_info:
    inertia = figure['inertia']

    if abs(inertia - reference_values['cap']['inertia']) < 50:
        print(
            f"Найдена фигура с моментом инерции {inertia} соответствует колпачку.")
    elif abs(inertia - reference_values['other_part']['inertia']) < 50:
        print(
            f"Найдена фигура с моментом инерции {inertia} соответствует другой детали.")

# Поиск кругов среди черных фигур
for figure in black_figures_info:
    contour = figure['contour']

    # Найдем окружности с помощью метода HoughCircles
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Рисуем окружности на изображении
            # Зеленый цвет для окружности
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Красный цвет для центра окружности
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# Отображение результата на изображении
for figure in black_figures_info:
    area = figure['area']
    center = figure['center']

    # Помечаем площадь и координаты
    text = f"Area: {area}"
    cv2.putText(image, text, (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Сохранение результата
cv2.imwrite('output_with_black_figures.jpg', image)

# Отображение результата
cv2.imshow('Detected Black Figures and Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
