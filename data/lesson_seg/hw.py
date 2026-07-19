import cv2
import numpy as np
from ultralytics import YOLO

# Завдання 1
# Відкрийте зображення data/lesson_seg/tumor1.jpg
# Проведіть сегментацію зображення використовуючи
# модель data/lesson_seg/brain-tumor-seg.jpg
# Визначте площу пухлини в пікселях.
# Визначте площу в
# (1 піксель – 0,0025
# )
# В залежності від площі присвойте пухлині певний тип
#  <10 – small
#  10-25 – middle
#  >25 – large
# Покажіть пухлину – за допомогою маски усі лишні
# пікселі зробіть 0, а як назву зображення використайте її тип

model = YOLO("brain-tumor-seg.pt")

img = cv2.imread("tumor1.jpg")

height, width = img.shape[:2]

results = model.predict(
    img,
    device="cpu"
)

result = results[0]

if result.masks is None:
    print("Пухлину не знайдено")
    exit()

masks = result.masks.data

max_area = 0
max_index = 0

for i in range(len(masks)):
    mask = masks[i].cpu().numpy().astype(np.uint8)

    area = int(np.sum(mask))

    if area > max_area:
        max_area = area
        max_index = i

area_pixels = max_area

area_real = area_pixels * 0.0025


if area_real < 10:
    tumor_type = "small"
elif area_real <= 25:
    tumor_type = "middle"
else:
    tumor_type = "large"

print(f"Площа в пікселях: {area_pixels}")
print(f"Площа: {area_real:.2f}")
print(f"Тип пухлини: {tumor_type}")

tumor_mask = masks[max_index].cpu().numpy().astype(np.uint8)

tumor_mask = cv2.resize(
    tumor_mask,
    (width, height)
)

mask = tumor_mask.astype(bool)

result_img = np.zeros_like(img)

result_img[mask] = img[mask]

cv2.imshow(tumor_type, result_img)

cv2.waitKey(0)
cv2.destroyAllWindows()