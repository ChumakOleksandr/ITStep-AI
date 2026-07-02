import cv2
import numpy as np

# Завдання 1
# Відкрийте зображення data/lesson3/notes.png. Проведіть
# наступні дії:
#  проведіть бінарізацію(звичайну та адаптивну)
#  застосуйте розмиття(гаусове) візьміть ядра 3, 5, 11 та
# sigmaX 0, 2, 10
#  повторіть бінарізацію, але перед тим застосуйте bilateral
# filter

# img = cv2.imread("notes.png")
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
# threshold = 128
#
# simple = gray.copy()
#
# mask = simple < threshold
#
# simple[mask] = 0
# simple[~mask] = 255
#
# cv2.imshow("simple", simple)
#
# adaptive = cv2.adaptiveThreshold(
#     gray,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     21,
#     2
# )
#
# cv2.imshow("adaptive", adaptive)
#
#
# gauss_3_0 = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)
# cv2.imshow("gauss 3 0", gauss_3_0)
#
# gauss_5_2 = cv2.GaussianBlur(gray, (5, 5), sigmaX=2)
# cv2.imshow("gauss 5 2", gauss_5_2)
#
# gauss_11_10 = cv2.GaussianBlur(gray, (11, 11), sigmaX=10)
# cv2.imshow("gauss 11 10", gauss_11_10)
#
#
# bilat = cv2.bilateralFilter(
#     gray,
#     d=9,
#     sigmaColor=75,
#     sigmaSpace=75
# )
#
# cv2.imshow("bilateral", bilat)
#
#
# simple_bilat = bilat.copy()
#
# mask = simple_bilat < threshold
#
# simple_bilat[mask] = 0
# simple_bilat[~mask] = 255
#
# cv2.imshow("simple bilateral", simple_bilat)
#
#
# adaptive_bilat = cv2.adaptiveThreshold(
#     bilat,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     21,
#     2
# )
#
# cv2.imshow("adaptive bilateral", adaptive_bilat)

# Завдання 2
# Відкрийте зображення data/lesson3/sudoku.jpg. Проведіть
# для нього бінарізацію, а саме
#  CLAHE
#  гаусове розмиття
#  адаптивна бінарізація
#  NLMean
# Самостійно підберіть параметри, збережіть результат.
# Порівняйте результати для гаусової та середньої адаптивної
# бінарізації

img = cv2.imread("sudoku.jpg")
cv2.imshow("orig", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

gray_clahe = clahe.apply(gray)

cv2.imshow("CLAHE", gray_clahe)


gauss = cv2.GaussianBlur(
    gray_clahe,
    (5, 5),
    sigmaX=2
)

cv2.imshow("Gaussian", gauss)

adaptive_gauss = cv2.adaptiveThreshold(
    gauss,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    6
)

cv2.imshow("Adaptive Gaussian", adaptive_gauss)

adaptive_mean = cv2.adaptiveThreshold(
    gauss,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    9,
    8
)

cv2.imshow("Adaptive Mean", adaptive_mean)

nlmean = cv2.fastNlMeansDenoising(
    gray_clahe,
    None,
    h=10,
    templateWindowSize=7,
    searchWindowSize=21
)

cv2.imshow("NLMean", nlmean)

adaptive_nlmean = cv2.adaptiveThreshold(
    nlmean,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

cv2.imshow("Adaptive NLMean", adaptive_nlmean)

cv2.imwrite(
    "sudoku_result.png",
    adaptive_nlmean
)



cv2.waitKey(0)
























