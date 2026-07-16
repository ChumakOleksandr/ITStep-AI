from ultralytics import YOLO
import numpy as np
import cv2

# Завдання 1
# Відкрийте зображення data/lesson_seg/crop3.jpg
# Проведіть сегментацію зображення використовуючи
# модель data/lesson_seg/crop-seg.jpg
# Покажіть усі маски рослин з підписами назви цієї
# рослини.
# Покажіть також самі рослини, для цього застосуйте
# маску, і всі зайві пікселі замініть на 255(зробити білий фон)

# model = YOLO("crop-seg.pt")
#
# img = cv2.imread("crop3.jpg")
#
# cv2.imshow("orig", img)
#
# height, width = img.shape[:2]
#
# results = model.predict(
#     img,
#     device="cpu"
# )
#
# result = results[0]
#
# cv2.imshow("result", result.plot())
#
# masks = result.masks
#
# boxes = result.boxes
#
# names = result.names
#
# masks_data = masks.data
#
# for i in range(len(masks_data)):
#
#     plant_mask = masks_data[i]
#
#     plant_mask = plant_mask.cpu().numpy()
#     plant_mask = plant_mask.astype(np.uint8)
#     plant_mask *= 255
#
#     plant_mask = cv2.resize(
#         plant_mask,
#         (width, height)
#     )
#
#     cls = int(boxes.cls[i].cpu().numpy())
#     class_name = names[cls]
#
#     cv2.imshow(
#         f"mask_{i}_{class_name}",
#         plant_mask
#     )
#
#     plant = np.ones_like(img) * 255
#
#     mask = plant_mask.astype(bool)
#
#     plant[mask] = img[mask]
#
#     cv2.imshow(
#         f"plant_{i}_{class_name}",
#         plant
#     )
#
#     print(f"Рослина {i}: {class_name}")

# Завдання 2
# Відкрийте зображення data/lesson_seg/crop3.jpg
# Проведіть сегментацію зображення
# Порахуйте розмір кожної рослини(площа маски)
# Покажіть найбільшу рослину кожного виду

model = YOLO("crop-seg.pt")

img = cv2.imread("crop3.jpg")

cv2.imshow("orig", img)

height, width = img.shape[:2]

results = model.predict(
    img,
    device="cpu"
)

result = results[0]

masks = result.masks
masks_data = masks.data

boxes = result.boxes

names = result.names

biggest_plants = {}

for i in range(len(masks_data)):

    plant_mask = masks_data[i]

    plant_mask = plant_mask.cpu().numpy()
    plant_mask = plant_mask.astype(np.uint8)

    area = int(np.sum(plant_mask))

    cls = int(boxes.cls[i].cpu().numpy())
    class_name = names[cls]

    print(
        f"Рослина {i}: "
        f"{class_name}, "
        f"площа = {area}"
    )

    if class_name not in biggest_plants:
        biggest_plants[class_name] = (area, i)
    elif area > biggest_plants[class_name][0]:
        biggest_plants[class_name] = (area, i)


for class_name, (area, index) in biggest_plants.items():

    plant_mask = masks_data[index]

    plant_mask = plant_mask.cpu().numpy()
    plant_mask = plant_mask.astype(np.uint8)
    plant_mask *= 255

    plant_mask = cv2.resize(
        plant_mask,
        (width, height)
    )

    mask = plant_mask.astype(bool)

    plant = np.ones_like(img) * 255

    plant[mask] = img[mask]

    cv2.imshow(
        f"largest_{class_name}",
        plant
    )

    print(
        f"Найбільша рослина {class_name}, "
        f"площа = {area}"
    )

cv2.waitKey(0)
cv2.destroyAllWindows()











































