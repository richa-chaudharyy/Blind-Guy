from ultralytics import YOLO
import cv2
import time

def detect_object(target_object):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    # Optional: reduce resolution to reduce memory usage
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Set desired frame interval (lower = faster)
    frame_interval = 0.5  # seconds --> means 2 FPS

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                if label.lower() == target_object.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    print(f"Detected {label} with confidence {conf}")

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # FPS Control
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_interval:
            time.sleep(frame_interval - elapsed_time)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_object("bottle")
