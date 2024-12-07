import cv2
import numpy as np
from main import check_black_cap


def test_black_cap_detected():
    black_cap_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(black_cap_image, (50, 50), 20, (255, 255, 255), -1)
    assert check_black_cap(black_cap_image)


def test_black_cap_not_detected():
    no_cap_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    assert not check_black_cap(no_cap_image)
