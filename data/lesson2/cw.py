import cv2
import numpy as np

# Завдання 1
# Відкрийте зображення data/lesson2/marbles.png.
# Використайте кольорову сегментацію для отримання масок до
# кульок:
#  синього кольору
#  зеленого і червоного
#  чорного
#  білого
#  усіх кульок

# img = cv2.imread("marbles.png")
#
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# def mouse(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print("HSV:", hsv[y, x])
#
# cv2.imshow("Image", img)
# cv2.setMouseCallback("Image", mouse)

# #  синього кольору
#
# lower_blue = (100, 70, 80)
# upper_blue = (130, 255, 255)
#
# mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
#
# cv2.imshow("Blue", mask_blue)
#
# #  зеленого і червоного
#
# lower_green = (40, 50, 50)
# upper_green = (80, 255, 255)
#
# mask_green = cv2.inRange(hsv, lower_green, upper_green)
# # cv2.imshow("Green", mask_green)
#
# lower_red = (0, 100, 150)
# upper_red = (7, 255, 255)
#
# mask_red = cv2.inRange(hsv, lower_red, upper_red)
# # cv2.imshow("Red", mask_red)
#
# mask_green_red = cv2.bitwise_or(mask_green, mask_red)
#
# cv2.imshow("Green + Red", mask_green_red)
#
#
#
# #  чорного
# lower_black = (0, 0, 0)
# upper_black = (100, 100, 40)
#
# mask_black = cv2.inRange(hsv, lower_black, upper_black)
#
# cv2.imshow("Black", mask_black)
#
# #  білого
#
# lower_white = (0, 0, 180)
# upper_white = (179, 50, 255)
#
# mask_white = cv2.inRange(hsv, lower_white, upper_white)
#
# cv2.imshow("White", mask_white)

# Завдання 2
# Відкрийте зображення data/lesson2/cell.png. Покращте
# зображення за допомогою вирівнювання гістограми. Оскільки
# зображення кольорове, вам доведеться зробити наступні
# кроки:
#  перевести зображення в LAB
#  розбити зображення на канали l, a та b
#  вирівняти гістограму для l
#  зібрати канали назад в зображення
#  перевести результат назад в BGR
# Порівняйте результати для 2 алгоритмів.

img = cv2.imread("cell.png")
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

l_eq = cv2.equalizeHist(l)

lab_eq = cv2.merge([l_eq, a, b])

result_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

cv2.imshow("Original", img)
cv2.imshow("EqualizeHist", result_eq)




cv2.waitKey(0)

