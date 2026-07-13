import cv2
import ultralytics

# Завдання 1
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та виведіть результат, підберіть
# параметри
# Можете змінити розмір кадру для кращої візуалізації
# cv2.resize()


# model = ultralytics.YOLO("yolo11s.pt")
#
# cap = cv2.VideoCapture("meetings.mp4")
#
# while True:
#     success, frame = cap.read()
#
#     if not success:
#         break
#
#
#     frame = cv2.resize(frame, None, fx=0.2, fy=0.2)
#
#
#     results = model.predict(
#         frame,
#         device="cuda",
#         conf=0.4,
#         iou=0.5
#     )
#
#     result = results[0]
#
#     res_img = result.plot()
#
#     cv2.imshow("Meetings Detection", res_img)
#
#     key = cv2.waitKey(30) & 0xFF
#
#     if key == ord('q'):
#         break
#
#     if key == ord(' '):
#         cv2.waitKey(0)
#
# cap.release()
# cv2.destroyAllWindows()

# Завдання 2
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та почніть показувати відео з
# моменту, коли людей стало 5

model = ultralytics.YOLO("yolo11s.pt")

cap = cv2.VideoCapture("meetings.mp4")

start_show = False

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, None, fx=0.2, fy=0.2)

    results = model.predict(
        frame,
        conf=0.6,
        iou=0.5
    )

    result = results[0]
    boxes = result.boxes

    people_count = 0

    for box in boxes:
        cls = int(box.cls.cpu().numpy()[0])

        if cls == 0:
            people_count += 1

    print(f"Людей: {people_count}")


    if people_count >= 5:
        start_show = True

    if start_show:
        cv2.imshow("Meetings", result.plot())

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()