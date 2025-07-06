import cv2
from ultralytics import YOLO
import pyttsx3
import time

tts_engine = pyttsx3.init()

def speak(text):
    print("Speaking:", text)
    tts_engine.say(text)
    tts_engine.runAndWait()

def object_navigation_mode(object_name, pretrained=True):
    if pretrained:
        model = YOLO("yolov8n.pt")  # pretrained yolov8 model
    else:
        model = YOLO("runs/detect/train/weights/best.pt")  # your custom model path

    cap = cv2.VideoCapture(0)

    detection_counter = 0
    detection_threshold = 3
    already_announced = False
    last_speak_time = time.time()
    speak_interval = 2  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        frame_center_x = frame.shape[1] // 2
        object_found = False

        for result in results:
            boxes = result.boxes
            names = result.names

            for box in boxes:
                class_id = int(box.cls[0])
                detected_label = names[class_id].lower()

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]

                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{detected_label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if object_name.lower() in detected_label:
                    detection_counter += 1
                    object_found = True

                    # Calculate object's horizontal position
                    object_center_x = (x1 + x2) // 2
                    offset = object_center_x - frame_center_x

                    if time.time() - last_speak_time > speak_interval:
                        if abs(offset) < 50:
                            speak(f"{object_name} is straight ahead.")
                        elif offset < 0:
                            speak(f"Turn slightly left to reach {object_name}.")
                        else:
                            speak(f"Turn slightly right to reach {object_name}.")
                        last_speak_time = time.time()

                else:
                    detection_counter = 0

        # After stable detection, announce final confirmation once
        if object_found and detection_counter >= detection_threshold and not already_announced:
            speak(f"{object_name} found successfully.")
            already_announced = True

        cv2.imshow("Object Navigation Mode", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
