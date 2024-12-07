import cv2
import numpy as np


def check_black_cap(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours) > 0  # Возвращает True, если черный колпачок найден


def main():
    camera = cv2.VideoCapture(0)  # Замените 0 на индекс вашей камеры

    if not camera.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Ошибка: Не удалось получить кадр.")
            break

        if check_black_cap(frame):
            print("Черный колпачок найден.")
            # Добавьте логику для дальнейшей обработки модели здесь
        else:
            print("Черный колпачок не найден. Удаляем компонент.")

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
