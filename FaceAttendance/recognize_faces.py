import cv2
import torch
import numpy as np
from pathlib import Path
from torch import nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# Import your attendance module
from mark_attendance import mark_attendance, get_person_name, PERSON_NAMES

# =====================================================
# SETTINGS
# =====================================================
MODEL_PATH = Path("models/face_cnn.pth")
IMAGE_SIZE = 128
CAMERA_INDEX = 2

CONFIDENCE_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MP_MODEL_DIR = Path("models/mp_models")
MP_MODEL_PATH = MP_MODEL_DIR / "blaze_face_short_range.tflite"
MP_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

# Track which faces we've already processed this session to avoid spamming the console
session_marked = set()


# =====================================================
# SAME CNN AS train_cnn.py
# =====================================================
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =====================================================
# LOAD TRAINED MODEL
# =====================================================
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}\nRun python train_cnn.py first.")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

if checkpoint.get("architecture") != "GrayscaleFaceCNN_v1":
    raise RuntimeError("Incompatible model. Run the matching train_cnn.py first.")

class_names = checkpoint["class_names"]
model = FaceCNN(len(class_names)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nModel loaded.")
print("Device:", DEVICE)
print("Registered identities:")
for label in class_names:
    category, person_id = label.split("_", 1)
    real_name = get_person_name(person_id)
    print(f" - {label}  ->  {real_name}")


# =====================================================
# FACE DETECTOR SETUP
# =====================================================
def download_mp_model(url, dest_path):
    if not dest_path.exists():
        print(f"Downloading MediaPipe model to {dest_path}...")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)

download_mp_model(MP_MODEL_URL, MP_MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=str(MP_MODEL_PATH))
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
face_detector = vision.FaceDetector.create_from_options(options)


# =====================================================
# CAMERA LOOP
# =====================================================
camera = cv2.VideoCapture(CAMERA_INDEX)
if not camera.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}.")

print("\nCamera started. Press 'Q' to quit.")
print(f"Attendance files saved to: {Path('attendance_records').resolve()}/")
frame_number = 0

try:
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_number += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = face_detector.detect(mp_image)
        detections = detection_result.detections

        status_text = ""

        if detections:
            for detection in detections:
                h, w, _ = frame.shape
                bbox = detection.bounding_box
                x, y = bbox.origin_x, bbox.origin_y
                bw, bh = bbox.width, bbox.height

                pad = 30
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.resize(face_gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                face_array = face_gray.astype(np.float32) / 255.0
                face_tensor = torch.from_numpy(face_array).unsqueeze(0).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(face_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]

                k = min(3, probabilities.numel())
                top_probs, top_indices = torch.topk(probabilities, k=k)

                best_score = top_probs[0].item()
                best_index = top_indices[0].item()
                best_label = class_names[best_index]
                second_score = top_probs[1].item() if k > 1 else 0.0
                margin = best_score - second_score

                if best_score >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
                    category, person_id = best_label.split("_", 1)
                    real_name = get_person_name(person_id)

                    # Only call mark_attendance once per session per person
                    if person_id not in session_marked:
                        marked = mark_attendance(person_id)
                        if marked:
                            session_marked.add(person_id)
                            status_text = f"Welcome, {real_name}!"
                            box_color = (0, 255, 0)  # Bright green for new
                        else:
                            status_text = f"{real_name} - Already marked today"
                            box_color = (0, 200, 0)  # Darker green for duplicate
                    else:
                        status_text = f"{real_name} - Present"
                        box_color = (0, 200, 0)

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # Draw name label
                    name_text = real_name
                    conf_text = f"{best_score:.0%}"

                    (tw1, th1), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    (tw2, th2), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    box_w = max(tw1, tw2) + 10
                    box_h = th1 + th2 + 15

                    cv2.rectangle(frame, (x1, y1 - box_h - 5), (x1 + box_w, y1 - 5), box_color, -1)
                    cv2.putText(frame, name_text, (x1 + 5, y1 - th2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, conf_text, (x1 + 5, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    status_text = ""

        if status_text:
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Attendance Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    face_detector.close()
    camera.release()
    cv2.destroyAllWindows()
    print("\nRecognition stopped.")