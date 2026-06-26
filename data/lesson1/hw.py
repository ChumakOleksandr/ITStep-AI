
import cv2
import numpy as np

# # Завдання 1
# # Відкрийте зображення data/Lenna.png. Прочитайте маски
# # data/mask1.png та data/mask2.png.
# # Об’єднайте дві маски в одну, скористайтесь cv2.bitwise_or()
# # та виведіть результат
# # Виведіть ту частину зображення, яка відповідає:
# #  mask1
# #  mask2
# #  mask1 і mask2
# # Усі пікселі які не відповідають маскам замінити на 0, перед
# # застосуванням змініть тип даних у масці на bool
# img = cv2.imread("Lenna.png", cv2.IMREAD_GRAYSCALE)
#
# mask1 = cv2.imread("mask1.png", cv2.IMREAD_GRAYSCALE)
# mask2 = cv2.imread("mask2.png", cv2.IMREAD_GRAYSCALE)
#
# #  mask1 і mask2
# mask_or = cv2.bitwise_or(mask1, mask2)
# mask_or_bool = mask_or == 255
#
# result3 = np.zeros_like(img)
# result3[mask_or_bool] = img[mask_or_bool]
#
# cv2.imshow("Mask OR", result3)
#
# #  mask1
# mask1_bool = mask1 == 255
# mask2_bool = mask2 == 255
# mask_or_bool = mask_or == 255
#
# result1 = np.zeros_like(img)
# result1[mask1_bool] = img[mask1_bool]
#
# cv2.imshow("Mask1 result", result1)
# #  mask2
# result2 = np.zeros_like(img)
# result2[mask2_bool] = img[mask2_bool]
#
# cv2.imshow("Mask2 result", result2)
# cv2.waitKey(0)

# Завдання 2
# Виведіть зображення. Підберіть самостійно межі

img = cv2.imread("baboo.jpg", cv2.IMREAD_GRAYSCALE)


eyes = img[10:50, 40:220]

cv2.imshow("Eyes", eyes)
cv2.waitKey(0)


