import cv2
import time
import os
import uuid
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import playsound
# USER SETTINGS
USER_LANGUAGE = "english"   # english |

LANG_MAP = {
    "english": "en",
    "hindi": "hi",
    "telugu": "te"
}

# LOAD YOLO MODEL
model = YOLO("yolov8n.pt")

# AUDIO FUNCTION

def speak(text):
    try:
        lang = LANG_MAP[USER_LANGUAGE]
        filename = f"audio_{uuid.uuid4()}.mp3"
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)
    except Exception as e:
        print("Audio error:", e)

# DIRECTION LOGIC
def get_direction(box, frame_width):    
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    if cx < frame_width * 0.4:
        return "left"
    elif cx > frame_width * 0.6:
        return "right"
    else:
        return "front"

# DISTANCE LOGIC

def estimate_distance(box, frame_area):
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    ratio = area / frame_area
    if ratio > 0.20:
        return "very close"
    elif ratio > 0.08:
        return "near"
    else:
        return "far"

# OBJECT PRIORITY
PRIORITY = {
    "car": 5, "bus": 5, "truck": 5,
    "motorcycle": 4, "bicycle": 3,
    "person": 3, "dog": 3,
    "stairs": 5,
    "chair": 2, "table": 2, "sofa": 2
}

# GUIDANCE ENGINE
def generate_guidance(obj, direction, distance):
    if obj in ["car", "bus", "truck"] and distance in ["near", "very close"]:
        return f"{obj} approaching, please stop"
    if obj == "stairs" and distance != "far":
        return "Stairs ahead, proceed carefully"
    if direction == "front" and distance in ["near", "very close"]:
        return "Obstacle ahead, move slightly right"
    if direction == "left":
        return "Obstacle on left, keep right"
    if direction == "right":
        return "Obstacle on right, keep left"
    return None

def is_path_clear(objects):
    for o in objects:
        if o["direction"] == "front" and o["distance"] in ["near", "very close"]:
            return False
    return True

# START WEBCAM
cap = cv2.VideoCapture(0)
last_spoken = ""
last_time = time.time()

print("System started. Press Q to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_area = h * w

    results = model(frame, stream=True)
    detected = []

    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < 0.5:
                continue

            cls_id = int(box.cls[0])
            obj = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            direction = get_direction((x1,y1,x2,y2), w)
            distance = estimate_distance((x1,y1,x2,y2), frame_area)
            priority = PRIORITY.get(obj, 1)

            detected.append({
                "obj": obj,
                "direction": direction,
                "distance": distance,
                "priority": priority
            })

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, obj, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    message = None
    if detected:
        detected.sort(key=lambda x: x["priority"], reverse=True)
        top = detected[0]
        message = generate_guidance(top["obj"], top["direction"], top["distance"])

    if message is None and is_path_clear(detected):
        message = "Path is clear, you may proceed"

    if message and message != last_spoken and time.time() - last_time > 3:
        speak(message)
        last_spoken = message
        last_time = time.time()

    cv2.imshow("AI Real-Time Guidance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


