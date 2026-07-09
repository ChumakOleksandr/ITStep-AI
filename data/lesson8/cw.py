import cv2
import ultralytics

# Завдання 1
# Отримайте перший кадр з файлу data\lesson8\animals.mp4
# та виведіть його на екран.
# Проведіть детекцію об’єктів зо допомогою YOLO та
# виведіть результати.
# Змініть параметри моделі conf та iou і подивіться як це
# впливає на результат.
# Отримайте рамки для кожного об’єкта, виріжіть їх та
# виведіть як окремі зображення

model = ultralytics.YOLO("yolo11s.pt")

cap = cv2.VideoCapture("animals.mp4")

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    results = model.predict(
        frame,
        device="cuda",
        conf=0.25,
        iou=0.5
    )

    result = results[0]

    cv2.imshow("Detection", result.plot())

    names = result.names
    boxes = result.boxes

    counter = {}

    for box in boxes:

        cls = int(box.cls.cpu().numpy()[0])
        class_name = names[cls]

        if class_name not in counter:
            counter[class_name] = 1
        else:
            counter[class_name] += 1

        obj_number = counter[class_name]

        xyxy = box.xyxy.cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy[0]

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            cv2.imshow(
                f"{class_name}_{obj_number}",
                roi
            )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
















