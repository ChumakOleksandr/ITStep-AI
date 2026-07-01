import cv2
import numpy as np


# Завдання 1
# Відкрийте зображення data\lesson2\darken.png. Проведіть з
# ним наступні операції, переведіть його в HSV формат та
# обробіть канал Value наступними способами:
#  застосуйте вирівнювання гістограм
#  збільшіть значення десь на 20-50%, оскільки тут
# результат буде типу float32 та явно вийде за межі [0-255]
# застосуйте np.clip(value, 0, 255) та value.astype(np.uint8)
# Виведіть результати обох обробок на екран

img = cv2.imread("darken.png")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

v_eq = cv2.equalizeHist(v)

hsv_eq = cv2.merge([h, s, v_eq])

result_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)



v_bright = v.astype(np.float32) * 1.5

v_bright = np.clip(v_bright, 0, 255)

v_bright = v_bright.astype(np.uint8)

hsv_bright = cv2.merge([h, s, v_bright])

result_bright = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)

cv2.imshow("Original", img)
cv2.imshow("EqualizeHist V", result_eq)
cv2.imshow("Value +50%", result_bright)

cv2.waitKey(0)
