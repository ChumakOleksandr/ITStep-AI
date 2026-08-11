import cv2
import ultralytics

# Завдання 1
# Відкрийте відео data/lesson_pose/sitting.mp4
# Отримайте перший кадр
# Покажіть його, за потреби змініть розмір

# video = cv2.VideoCapture("sitting.mp4")
#
# ret, frame = video.read()
#
# if ret:
#     frame = cv2.resize(
#         frame,
#         (800, 600)
#     )
#
#     cv2.imshow("First Frame", frame)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Не вдалося зчитати кадр")
#
# video.release()

# Завдання 2
# Застосуйте модель YOLO Pose
# Отримайте результати (result) та виведіть їх на екран
# Використайте параметри device

# model = ultralytics.YOLO("yolo11s-pose.pt")
#
# video = cv2.VideoCapture("sitting.mp4")
#
# ret, frame = video.read()
#
# if ret:
#     results = model.predict(
#         frame,
#         device="cpu",
#         conf=0.5
#     )
#
#     result = results[0]
#
#     print(result)
#
#     res_img = result.plot()
#
#     cv2.imshow("result", res_img)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# else:
#     print("Не вдалося зчитати кадр")
#
# video.release()

# Завдання 3
# Користуючись методом plot() отримайте зображення з
# рамками та підписами і покажіть його.

# model = ultralytics.YOLO("yolo11s-pose.pt")
#
# video = cv2.VideoCapture("sitting.mp4")
#
# ret, frame = video.read()
#
# if ret:
#     results = model.predict(
#         frame,
#         device="cpu",
#         conf=0.5
#     )
#
#     result = results[0]
#
#     res_img = result.plot()
#
#     cv2.imshow("result", res_img)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# video.release()

# Завдання 4
# ● Отримайте інформацію про ключові точки(keypoints)
# ● Виведіть її на екран
# ● Отримайте координати точок(xy)
# ● Виведіть координати на екран разом з типом даних та
# розміром(позбудьтесь тензорів за допомогою cpu() та
# numpy())

# model = ultralytics.YOLO("yolo11s-pose.pt")
#
# video = cv2.VideoCapture("sitting.mp4")
#
# ret, frame = video.read()
#
# if ret:
#     results = model.predict(
#         frame,
#         device="cpu",
#         conf=0.5
#     )
#
#     result = results[0]
#
#     keypoints = result.keypoints
#
#     print("KEYPOINTS:")
#     print(keypoints)
#
#     xy = keypoints.xy
#
#     print("\nXY (tensor):")
#     print(xy)
#     print("Тип:", type(xy))
#     print("Розмір:", xy.shape)
#
#     xy = xy.cpu().numpy()
#
#     print("\nXY (numpy):")
#     print(xy)
#     print("Тип:", type(xy))
#     print("Розмір:", xy.shape)
#
# else:
#     print("Не вдалося зчитати кадр")
#
# video.release()

# Завдання 5
# ● Отримайте координати для лівого коліна, лівої руки,
# правої руки для першого об’єкта
# ● Намалюйте ці точки на зображенні:
# ○ ліве коліно – зелений
# ○ ліва рука – червоний
# ○ права рука – білий

# model = ultralytics.YOLO("yolo11s-pose.pt")
#
# video = cv2.VideoCapture("sitting.mp4")
#
# ret, img = video.read()
#
# if ret:
#     results = model.predict(
#         img,
#         device="cpu",
#         conf=0.5
#     )
#
#     result = results[0]
#
#     xy = result.keypoints.xy
#
#     xy = xy.cpu().numpy()
#
#     xy = xy[0]
#
#     xy = xy.astype(int)
#
#     x_left_knee, y_left_knee = xy[13]
#
#     x_left_hand, y_left_hand = xy[9]
#
#     x_right_hand, y_right_hand = xy[10]
#
#     cv2.circle(
#         img,
#         (x_left_knee, y_left_knee),
#         15,
#         (0, 255, 0),
#         -1
#     )
#
#     cv2.circle(
#         img,
#         (x_left_hand, y_left_hand),
#         15,
#         (0, 0, 255),
#         -1
#     )
#
#     cv2.circle(
#         img,
#         (x_right_hand, y_right_hand),
#         15,
#         (255, 255, 255),
#         -1
#     )
#
#     cv2.putText(
#         img,
#         "Left Knee",
#         (x_left_knee + 10, y_left_knee),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         (0, 255, 0),
#         2
#     )
#
#     cv2.putText(
#         img,
#         "Left Hand",
#         (x_left_hand + 10, y_left_hand),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         (0, 0, 255),
#         2
#     )
#
#     cv2.putText(
#         img,
#         "Right Hand",
#         (x_right_hand + 10, y_right_hand),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         (255, 255, 255),
#         2
#     )
#
#     cv2.imshow("result", img)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# else:
#     print("Не вдалося зчитати кадр")
#
# video.release()

# Завдання 6
# Для кожного кадру на відео намалюйте координати для
# лівого коліна, лівої руки, правої руки
# Беріть координати для першого об’єкта

# model = ultralytics.YOLO("yolo11s-pose.pt")
#
# video = cv2.VideoCapture("sitting.mp4")
#
# while True:
#     ret, img = video.read()
#
#     if not ret:
#         break
#
#     results = model.predict(
#         img,
#         device="cpu",
#         conf=0.5,
#         verbose=False
#     )
#
#     result = results[0]
#
#     if result.keypoints is not None:
#
#         xy = result.keypoints.xy.cpu().numpy()
#
#         if len(xy) > 0:
#
#             xy = xy[0].astype(int)
#
#             x_left_hand, y_left_hand = xy[9]
#
#             x_right_hand, y_right_hand = xy[10]
#
#             x_left_knee, y_left_knee = xy[13]
#
#             cv2.circle(
#                 img,
#                 (x_left_knee, y_left_knee),
#                 10,
#                 (0, 255, 0),
#                 -1
#             )
#
#             cv2.circle(
#                 img,
#                 (x_left_hand, y_left_hand),
#                 10,
#                 (0, 0, 255),
#                 -1
#             )
#
#             cv2.circle(
#                 img,
#                 (x_right_hand, y_right_hand),
#                 10,
#                 (255, 255, 255),
#                 -1
#             )
#
#     cv2.imshow("Pose", img)
#
#     key = cv2.waitKey(25)
#     if key == 27:
#         break
#
# video.release()
# cv2.destroyAllWindows()

# Завдання 7
# Під час відео обраховуйте кількість присідань.
# Вважайте що людина присіла якщо рука опустилась
# нижче коліна.
# Кількість присідань відображайте на кадрі(cv2.putText)

# model = ultralytics.YOLO("yolo11s-pose.pt")
#
# video = cv2.VideoCapture("sitting.mp4")
#
# squat_count = 0
# is_sitting = False
#
# while True:
#     ret, img = video.read()
#
#     if not ret:
#         break
#
#     results = model.predict(
#         img,
#         device="cpu",
#         conf=0.5,
#         verbose=False
#     )
#
#     result = results[0]
#
#     if result.keypoints is not None:
#
#         xy = result.keypoints.xy.cpu().numpy()
#
#         if len(xy) > 0:
#             xy = xy[0].astype(int)
#
#             x_left_hand, y_left_hand = xy[9]
#
#             x_left_knee, y_left_knee = xy[13]
#
#             cv2.circle(
#                 img,
#                 (x_left_hand, y_left_hand),
#                 10,
#                 (0, 0, 255),
#                 -1
#             )
#
#             cv2.circle(
#                 img,
#                 (x_left_knee, y_left_knee),
#                 10,
#                 (0, 255, 0),
#                 -1
#             )
#
#             if y_left_hand > y_left_knee:
#
#                 if not is_sitting:
#                     squat_count += 1
#                     is_sitting = True
#
#             else:
#                 is_sitting = False
#
#     cv2.putText(
#         img,
#         f"Squats: {squat_count}",
#         (30, 50),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (255, 255, 255),
#         2
#     )
#
#     cv2.imshow("Squat Counter", img)
#
#     if cv2.waitKey(25) == 27:
#         break
#
# video.release()
# cv2.destroyAllWindows()

# Завдання 8
# Модифікуйте код щоб кількість присідань виводилась
# правильно. Для цього вам потрібно визначати чи людина
# зараз присідає чи піднімається за правилом:
# ● якщо рука нижче коліна то людина встає
# ● якщо рука вище коліна – присідає
# Рахуйте лише ті присідання які відбулись коли людина
# присідає та рука опинилась нижче коліна.
# Разом з кількістю присідань відображайте чи людина
# присідає чи встає
#
# model = ultralytics.YOLO("yolo11s-pose.pt")
#
# video = cv2.VideoCapture("sitting.mp4")
#
# squat_count = 0
#
# state = "Присідає"
# counted = False
#
# while True:
#     ret, img = video.read()
#
#     if not ret:
#         break
#
#     results = model.predict(
#         img,
#         device="cpu",
#         conf=0.5,
#         verbose=False
#     )
#
#     result = results[0]
#
#     if result.keypoints is not None:
#
#         xy = result.keypoints.xy.cpu().numpy()
#
#         if len(xy) > 0:
#             xy = xy[0].astype(int)
#
#             x_hand, y_hand = xy[9]
#
#             x_knee, y_knee = xy[13]
#
#             cv2.circle(
#                 img,
#                 (x_hand, y_hand),
#                 10,
#                 (0, 0, 255),
#                 -1
#             )
#
#             cv2.circle(
#                 img,
#                 (x_knee, y_knee),
#                 10,
#                 (0, 255, 0),
#                 -1
#             )
#
#             if y_hand < y_knee:
#                 state = "squats down"
#                 counted = False
#
#             else:
#                 state = "gets up"
#
#                 if not counted:
#                     squat_count += 1
#                     counted = True
#
#     cv2.putText(
#         img,
#         f"Squats: {squat_count}",
#         (20, 50),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (255, 255, 255),
#         2
#     )
#
#     cv2.putText(
#         img,
#         state,
#         (20, 100),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 255),
#         2
#     )
#
#     cv2.imshow("Squat Counter", img)
#
#     if cv2.waitKey(25) == 27:
#         break
#
# video.release()
# cv2.destroyAllWindows()

# Завдання 9
# ● Отримайте 258 кадр з відео
# ● Застосуйте модель
# ● Отримайте результати(result)
# ● Отримайте дані про рамки(boxes)
# ● Отримайте дані про рамку для першого об’єкта та
# виведіть їх
# ● Відобразіть результати(метод plot())
# ● Зробіть висновки

model = ultralytics.YOLO("yolo11s-pose.pt")

video = cv2.VideoCapture("sitting.mp4")

video.set(cv2.CAP_PROP_POS_FRAMES, 258)

ret, img = video.read()

if ret:
    results = model.predict(
        img,
        device="cpu"
    )

    result = results[0]

    boxes = result.boxes

    print("Усі рамки:")
    print(boxes)

    first_box = boxes[0]

    print("\nПерша рамка:")
    print(first_box)

    print("\nКоординати рамки xyxy:")
    print(first_box.xyxy)

    print("\nКлас:")
    print(first_box.cls)

    print("\nЙмовірність:")
    print(first_box.conf)

    res_img = result.plot()

    cv2.imshow("Result", res_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

video.release()