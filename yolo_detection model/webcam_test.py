from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")
cap = cv2.VideoCapture(0, cv2.CAP_ANY)

if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    exit()

print("Webcam opened! Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame!")
        break

    results = model.predict(frame, conf=0.25)
    annotated = results[0].plot()
    cv2.imshow("YOLO Webcam Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
