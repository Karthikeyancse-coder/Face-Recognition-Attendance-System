
import cv2
import os
import csv
import re
import time
from pathlib import Path

# ---------------- CONFIGURATION ----------------
DATASET_DIR = Path("dataset")
TARGET_IMAGES = 50       # Increase to 300 after testing
CAMERA_INDEX = 2
IMAGE_SIZE = (128, 128)

STAFF_IDS = [f"ST00{i}" for i in range(1, 3)]
STUDENT_IDS = [f"SD00{i}" for i in range(1, 11)]

# OpenCV's built-in Haar face detector
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load the face detector.")

# ---------------- SELECT PERSON ----------------
print("\n=== FACE DATASET COLLECTION ===")
print("1. Staff")
print("2. Student")

category_choice = input("Select category (1/2): ").strip()

if category_choice == "1":
    category = "staff"
    valid_ids = STAFF_IDS
elif category_choice == "2":
    category = "students"
    valid_ids = STUDENT_IDS
else:
    raise SystemExit("Invalid category selection.")

print("\nAvailable IDs:")
print(", ".join(valid_ids))

person_id = input("Enter the person's ID: ").strip().upper()

if person_id not in valid_ids:
    raise SystemExit("Invalid ID for the selected category.")

person_name = input("Enter the person's name: ").strip()

if not person_name or not re.fullmatch(
    r"[A-Za-z0-9 _-]+", person_name
):
    raise SystemExit("Enter a valid name using English letters.")

# Create a separate folder for this person
person_dir = DATASET_DIR / category / person_id
person_dir.mkdir(parents=True, exist_ok=True)

# ---------------- OPEN CAMERA ----------------
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise SystemExit(
        "Could not open webcam. Check camera permissions "
        "or try CAMERA_INDEX = 1."
    )

# Use a CSV file to record image labels
metadata_file = DATASET_DIR / "metadata.csv"
metadata_file.parent.mkdir(parents=True, exist_ok=True)

file_exists = metadata_file.exists()

# Count existing images so collection can be resumed
existing_images = list(person_dir.glob("*.jpg"))
saved_count = len(existing_images)

print(f"\nPerson: {person_name} ({person_id})")
print(f"Images already saved: {saved_count}")
print(f"Target: {TARGET_IMAGES}")
print("\nLook at the camera and change your pose naturally.")
print("Press S to save a detected face.")
print("Press Q to quit.")

last_capture_time = 0

try:
    while saved_count < TARGET_IMAGES:
        success, frame = camera.read()

        if not success:
            print("Could not read from webcam.")
            break

        # Mirror the preview for easier positioning
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # Select the largest detected face
        selected_face = None

        if len(faces) > 0:
            selected_face = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            x, y, w, h = selected_face
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

        cv2.putText(
            frame,
            f"{person_id}: {saved_count}/{TARGET_IMAGES}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        message = (
            "Press S to save face"
            if selected_face is not None
            else "No face detected"
        )

        cv2.putText(
            frame, message, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 2
        )

        cv2.imshow("Face Dataset Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s") and selected_face is not None:
            # Avoid saving images too quickly
            if time.time() - last_capture_time < 0.4:
                continue

            x, y, w, h = selected_face

            # Add a margin around the detected face
            margin = int(0.20 * max(w, h))
            center_x = x + w // 2
            center_y = y + h // 2
            side = max(w, h) + 2 * margin

            left = max(0, center_x - side // 2)
            top = max(0, center_y - side // 2)
            right = min(frame.shape[1], left + side)
            bottom = min(frame.shape[0], top + side)

            face_crop = frame[top:bottom, left:right]

            if face_crop.size == 0:
                continue

            # Resize each face to a consistent size
            face_crop = cv2.resize(
                face_crop, IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )

            saved_count += 1
            image_name = f"{person_id}_{saved_count:04d}.jpg"
            image_path = person_dir / image_name

            if not cv2.imwrite(str(image_path), face_crop):
                saved_count -= 1
                print("Could not save image.")
                continue

            # Append the image label to the metadata CSV
            with open(
                metadata_file, "a", newline="", encoding="utf-8"
            ) as csv_file:
                writer = csv.writer(csv_file)

                if not file_exists:
                    writer.writerow([
                        "image_path", "person_id",
                        "person_name", "category"
                    ])
                    file_exists = True

                writer.writerow([
                    str(image_path),
                    person_id,
                    person_name,
                    category
                ])

            last_capture_time = time.time()
            print(f"Saved {saved_count}/{TARGET_IMAGES}: {image_name}")

    if saved_count >= TARGET_IMAGES:
        print("\nTarget reached!")

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"\nImages saved in: {person_dir}")
print(f"Total images for this person: {saved_count}")
print("Dataset collection session finished.")
