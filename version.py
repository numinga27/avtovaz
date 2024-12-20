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
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Список для хранения информации о черных фигурах
black_figures_info = []

print(f"Найдено контуров: {len(contours)}")

# Обработка контуров черных фигур
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    print(f"Обрабатываем контур {i}: Площадь = {area}")

    if area < 100:  # Фильтрация по площади (можно настроить)
        print("Контур пропущен из-за маленькой площади.")
        continue

    # Вычисление центра масс
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Вычисление момента инерции
    inertia = cv2.contourArea(contour) * (cX**2 + cY**2)

    # Сохранение информации о черной фигуре
    black_figures_info.append({
        'number': i + 1,
        'area': area,
        'center': (cX, cY),
        'inertia': inertia,
        'contour': contour
    })

# Поиск колпачка или круга по критериям площади и форме
for figure in black_figures_info:
    area = figure['area']
    center = figure['center']
    
    # Условие для определения колпачка или круга (можно настроить)
    if area > 1000:  # Примерный порог для колпачка
        print(f"Найдена фигура {figure['number']} как колпачок/круг с площадью: {area}")

        # Подписываем фигуру на изображении
        text = f"Cap #{figure['number']} Area: {area} Inertia: {figure['inertia']}"
        cv2.putText(image, text, (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Сохранение информации в файл
with open('black_figures_info.txt', 'w') as f:
    for figure in black_figures_info:
        f.write(f"Figure #{figure['number']}: Area = {figure['area']}, Inertia = {figure['inertia']}\n")

# Отображение результата на изображении
cv2.imshow('Detected Black Figures', image)

# Сохранение результата
cv2.imwrite('output_with_black_figures.jpg', image)

# Запрос у пользователя детали по площади
user_area = float(input("Введите площадь детали для поиска: "))
detail_name = input("Введите название детали: ")

# Сохранение информации о детали в файл
with open('details_info.txt', 'a') as f:
    f.write(f"Detail Name: {detail_name}, Area: {user_area}\n")

print("Информация о детали сохранена.")

cv2.waitKey(0)
cv2.destroyAllWindows()

