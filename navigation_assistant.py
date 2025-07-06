import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Text-to-Speech engine
tts_engine = pyttsx3.init()

def speak(text):
    print("Speaking:", text)
    tts_engine.say(text)
    tts_engine.runAndWait()

def navigation_mode():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    zone_width = frame_width // 3

    # Expanded obstacle list (can modify as needed)
    known_obstacles = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "bench", "train"
    ]

    last_speak_time = time.time()
    speak_interval = 2  # seconds
    pole_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        left, center, right = False, False, False
        pole_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                box_center = (x1 + x2) // 2
                if box_center < zone_width:
                    left = True
                elif box_center < 2 * zone_width:
                    center = True
                else:
                    right = True

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Pole detection logic (heuristic: narrow tall object)
                width = x2 - x1
                height = y2 - y1
                if width < 60 and height > 350:
                    pole_detected = True

        # Speech only after interval
        if time.time() - last_speak_time > speak_interval:
            if center:
                speak("Obstacle ahead")
            elif left and not right:
                speak("Turn slightly right")
            elif right and not left:
                speak("Turn slightly left")
            elif left and right:
                speak("Both sides blocked, proceed carefully")
            else:
                speak("Path clear")

            if pole_detected:
                speak("Pole ahead")

            last_speak_time = time.time()

        # Draw navigation zones (optional visualization)
        cv2.line(frame, (zone_width, 0), (zone_width, frame.shape[0]), (255, 255, 0), 2)
        cv2.line(frame, (2 * zone_width, 0), (2 * zone_width, frame.shape[0]), (255, 255, 0), 2)

        cv2.imshow("Outdoor Navigation", frame)

        # Quit on pressing Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Add tiny delay for smoother loop
        time.sleep(0.2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    navigation_mode()
