import cv2

# Завдання 1
# Відкрийте відео з файлу data\lesson7\meter.mp4.
# Проведіть бінарізацію кадрів та збережіть в новий файл.
# Можливо очистіть від шуму або наведіть різкість через
# bilateralFilter

cap = cv2.VideoCapture("meter.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out_writer = cv2.VideoWriter(
    "meter_result.mp4",
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

    sharp = cv2.addWeighted(
        gray,  # оригінал
        1.5,  # вага оригіналу
        cv2.GaussianBlur(gray, (0, 0), 3),  # розмите зображення
        -0.5,  # віднімаємо його
        0
    )

    bilat = cv2.bilateralFilter(
        sharp,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    adapt = cv2.adaptiveThreshold(
        bilat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    cv2.imshow("adapt", adapt)

    out_writer.write(adapt)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out_writer.release()
cap.release()
cv2.destroyAllWindows()

