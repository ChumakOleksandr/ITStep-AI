
import cv2
import numpy as np

# Завдання 1
# Відкрийте зображення data/lesson3/sonet.png. Проведіть
# бінарізацію.
# Обов’язково використайте:
#  розмиття або наведення різкості
#  адаптивну бінарізацію
#  очищеня шумів

# img = cv2.imread("sonet.png")
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("gray", gray)
#
# gauss = cv2.GaussianBlur(
#     gray,
#     (3, 3),
#     sigmaX=2
# )
#
# cv2.imshow("gauss", gauss)
#
# binary = cv2.adaptiveThreshold(
#     gauss,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     21,
#     3
# )
#
# cv2.imshow("binary", binary)
#
# kernel = np.ones((3,3), dtype=np.uint8)
#
# clean = cv2.morphologyEx(
#     binary,
#     cv2.MORPH_OPEN,
#     kernel
# )
#
# cv2.imshow("clean", clean)

# Завдання 2
# Відкрийте зображення data/lesson3/sonnet_noised.png.
# Проведіть бінарізацію. Застосуйте код з завдання 1 та
# спробуйте покращити результат


img = cv2.imread("sonet_noised.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)

nlmean = cv2.fastNlMeansDenoising(
    gray,
    None,
    h=15,
    templateWindowSize=3,
    searchWindowSize=9
)

cv2.imshow("nlmean", nlmean)


gauss = cv2.GaussianBlur(
    nlmean,
    (3, 3),
    sigmaX=1
)

cv2.imshow("gauss", gauss)

binary = cv2.adaptiveThreshold(
    gauss,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

cv2.imshow("binary", binary)

kernel = np.ones((3, 3), dtype=np.uint8)

clean = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

cv2.imshow("clean", clean)


cv2.waitKey(0)
