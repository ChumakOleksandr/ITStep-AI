import cv2
import ultralytics
import utils

# Завдання 1
# Відкрийте відео data/lesson_pose/squat.mp4
# Ваша задача рахувати кількість присідань.
# Отримайте перший кадр та виділіть основні точки.
# Отримайте координати 3-ох точок ноги
# Визначте кут між цими трьома точками. Скористайтесь
# функцією utils.get_angle(x1, y1, x2, y2, x3, y3) де x2, y2 –
# координати коліна(центральна точка)
# Запустіть відео та добавте на сам кадр кут згинання ніг.
# Визначіть нижню межу кута(якщо людина опустилась
# нижче вважаємо що вона достатньо опустилась) та верхню
# межу кута(якщо людина піднялась вище вважаємо що вона
# достатньо піднялась)
# Добавте кількість присідань та
# кут на кожен кадр.


model = ultralytics.YOLO("yolo11s-pose.pt")

video = cv2.VideoCapture(
    "squat.mp4"
)

squat_count = 0

is_down = False

LOWER_ANGLE = 90
UPPER_ANGLE = 160

while True:

    ret, img = video.read()

    if not ret:
        break

    results = model.predict(
        img,
        device="cpu",
        conf=0.5,
        verbose=False
    )

    result = results[0]

    if result.keypoints is not None:

        xy = result.keypoints.xy.cpu().numpy()

        if len(xy) > 0:

            xy = xy[0].astype(int)

            x1, y1 = xy[11]

            x2, y2 = xy[13]

            x3, y3 = xy[15]

            angle = utils.get_angle(
                x1, y1,
                x2, y2,
                x3, y3
            )

            cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), -1)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), -1)

            cv2.putText(
                img,
                f"{int(angle)}",
                (x2 + 15, y2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            if angle < LOWER_ANGLE:
                is_down = True

            if angle > UPPER_ANGLE and is_down:
                squat_count += 1
                is_down = False

    cv2.putText(
        img,
        f"Squats: {squat_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    img_show = cv2.resize(
        img,
        (1000, 700)
    )

    cv2.imshow("Squat Counter", img_show)

    if cv2.waitKey(25) == 27:
        break

video.release()
cv2.destroyAllWindows()