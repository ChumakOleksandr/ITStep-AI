import cv2
import numpy as np

# # Завдання 1
# # Виведіть відео з файлу data\lesson7\text.mp4 на екран та
# # збережіть в новий файл.
# # Змініть розмір зображення.
#
# cap = cv2.VideoCapture("text.mp4")
#
# fps = int(cap.get(cv2.CAP_PROP_FPS))
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#
# out_writer = cv2.VideoWriter(
#     "result.mp4",
#     fourcc,
#     fps,
#     (500, 500)
# )
#
# while True:
#
#     success, frame = cap.read()
#
#     if not success:
#         break
#
#     new_frame = cv2.resize(
#         frame,
#         (500, 500)
#     )
#
#     cv2.imshow("new_frame", new_frame)
#
#     out_writer.write(new_frame)
#
#     if cv2.waitKey(40) & 0xFF == ord('q'):
#         break
#
# out_writer.release()
# cap.release()
# cv2.destroyAllWindows()

# Завдання 2
# Відкрийте відео з файлу data\lesson7\text.mp4. Проведіть
# бінарізацію кадрів та збережіть в новий файл.

cap = cv2.VideoCapture("result.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out_writer = cv2.VideoWriter(
    "binary_video.mp4",
    fourcc,
    fps,
    (width, height),
    isColor=False
)

while True:

    success, frame = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    threshold = 128

    binary = gray.copy()

    mask = binary < threshold

    binary[mask] = 0
    binary[~mask] = 255

    cv2.imshow("binary", binary)

    out_writer.write(binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out_writer.release()
cap.release()
cv2.destroyAllWindows()
