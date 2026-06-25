import numpy as np
import cv2

# Завдання 1
# Відкрийте зображення data/Lenna.png. Виведіть на екран
# розмір зображення, тип даних, максимальну та мінімальну
# інтенсивність пікселів, саме зображення з підписом.

# img = cv2.imread(
#     "Lenna.png",
#       cv2.IMREAD_GRAYSCALE,
# )
#
# if img is None:
#     print("Зображення не знайдено!")
# else:
#     print("Shape:", img.shape)
#     print("Dtype:", img.dtype)
#     print("Min pixel value:", np.min(img))
#     print("Max pixel value:", np.max(img))
#
#     cv2.imshow("Lenna Image", img)
#     cv2.waitKey(0)

# Завдання 2
# Відкрийте зображення data/Lenna.png. Виведіть на екран
# такі зображень:
#  Верхній лівий кут розміром 100х50
#  Центральний квадрат розміром 100х100
#  Верхню половину
#  Нижню половину
#  Ліву половину
#  Праву половину

# h, w = img.shape
#
# top_left = img[0:100, 0:50]
#
# center_y = h // 2
# center_x = w // 2
# center_square = img[center_y - 50:center_y + 50,
#                     center_x - 50:center_x + 50]
#
# top_half = img[:h // 2, :]
#
# bottom_half = img[h // 2:, :]
#
# left_half = img[:, :w // 2]
#
# right_half = img[:, w // 2:]
#
# cv2.imshow("Original", img)
# cv2.imshow("Top Left (100x50)", top_left)
# cv2.imshow("Center 100x100", center_square)
# cv2.imshow("Top Half", top_half)
# cv2.imshow("Bottom Half", bottom_half)
# cv2.imshow("Left Half", left_half)
# cv2.imshow("Right Half", right_half)
#
# cv2.waitKey(0)

# Завдання 3
# Відкрийте зображення data/Lenna.png. Створіть наступні
# зображення

# h, w = img.shape
#
# # Перше зображення
# img1 = img.copy()
# img1[0:20, :] = 0
# img1[236:256, :] = 255
# cv2.imshow("Перше", img1)
#
# # друге зображення
#
# img2 = img.copy()
# img2[:, 0:20] = 0
# img2[:, 236:256] = 0
# cv2.imshow("Друге", img2)
#
# # третє зображення
# img3 = np.zeros((256, 256), dtype=np.uint8)
# img3[50:206, 50:206] = img[50:206, 50:206]
# cv2.imshow("Третє", img3)


# Завдання 4
# Відкрийте зображення data/Lenna.png. Створіть маску для
# пік селів з інтенсивністю більше 128 та виведіть її. Також
# виведіть заперечення цієї маски.
# На оригінальному зображенні, усі пікселі які не
# відповідають масці замініть на 0 та виведіть результат

mask = img > 128

print("Mask:")
print(mask)

inv_mask = ~mask

print("Inverted mask:")
print(inv_mask)

result = img.copy()
result[inv_mask] = 0

cv2.imshow("Result", result)
cv2.waitKey(0)

