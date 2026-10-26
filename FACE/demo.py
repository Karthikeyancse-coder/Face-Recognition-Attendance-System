"""
Webcam-based Multi-Person Face Verification using ArcFace (InsightFace buffalo_l)
----------------------------------------------------------------------------------
Step 1: Enroll one or more people using a few webcam captures each, tagged with a name
Step 2: Run live webcam verification that shows the matched person's name

Usage:
    python webcam_face_verify.py enroll <name>     # e.g. python webcam_face_verify.py enroll Willson
    python webcam_face_verify.py verify             # run live verification against all enrolled people
    python webcam_face_verify.py list                # list currently enrolled people
    python webcam_face_verify.py delete <name>      # remove a person from the database
"""

import sys
import os
import time
import pickle
import numpy as np
import cv2

# Windows DLL path resolution for CUDA / cuDNN
if sys.platform == "win32":
    dll_directories = []
    cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
    cuda_bin = os.path.join(cuda_path, "bin")
    if os.path.exists(cuda_bin) and hasattr(os, "add_dll_directory"):
        dll_directories.append(os.add_dll_directory(cuda_bin))
    torch_dll = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_dll) and hasattr(os, "add_dll_directory"):
        dll_directories.append(os.add_dll_directory(torch_dll))
        os.environ["PATH"] = torch_dll + os.pathsep + os.environ.get("PATH", "")

from insightface.app import FaceAnalysis

ENROLLED_PATH = "enrolled.pkl"
THRESHOLD = 0.5  # tune this after testing on your own data
NUM_ENROLL_SHOTS = 5

# --- Verification lock settings ---
AUTO_RESET_SECONDS = None  # e.g. 5 to auto-reset after 5s, or None to require 'r'


def load_model():
    """Load ArcFace (buffalo_l) with GPU acceleration."""
    # ctx_id=0 targets the primary GPU (RTX 5070)
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding(app, frame):
    """Detect face(s) in a frame and return the embedding of the largest face."""
    faces = app.get(frame)
    if len(faces) == 0:
        return None, None
    # Pick largest detected face by bounding box area
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].embedding, faces[0].bbox


def load_database():
    """Load the enrolled-people database: { name: embedding }."""
    if not os.path.exists(ENROLLED_PATH):
        return {}
    with open(ENROLLED_PATH, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, np.ndarray):
        return {"person": data}
    return data


def save_database(db):
    with open(ENROLLED_PATH, "wb") as f:
        pickle.dump(db, f)


def open_camera(index=0):
    """Open camera using DirectShow backend on Windows, default on other platforms."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    return cap


def enroll(app, name):
    """Capture several webcam shots for `name` and save the averaged embedding."""
    cap = open_camera(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    embeddings = []
    print(f"Enrolling '{name}'. Press SPACE to capture a shot ({NUM_ENROLL_SHOTS} needed). Press Q to cancel.")

    while len(embeddings) < NUM_ENROLL_SHOTS:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(
            display,
            f"{name}: {len(embeddings)}/{NUM_ENROLL_SHOTS}  [SPACE=capture, Q=quit]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Enrollment", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            emb, bbox = get_embedding(app, frame)
            if emb is None:
                print("No face detected, try again.")
                continue
            embeddings.append(emb)
            print(f"Captured shot {len(embeddings)}/{NUM_ENROLL_SHOTS}")
        elif key == ord("q"):
            print("Enrollment cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        print("No embeddings captured, enrollment aborted.")
        return

    avg_embedding = np.mean(embeddings, axis=0)

    db = load_database()
    if name in db:
        print(f"'{name}' already enrolled — overwriting with new embedding.")
    db[name] = avg_embedding
    save_database(db)

    print(f"Enrollment complete for '{name}'. Database now has {len(db)} people: {', '.join(db.keys())}")


def identify(embedding, db):
    """Compare an embedding against every enrolled person and return the best match."""
    best_name = None
    best_score = -1.0
    for name, ref_embedding in db.items():
        score = cosine_similarity(embedding, ref_embedding)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score


def verify(app):
    """Run live webcam verification against all enrolled identities."""
    db = load_database()
    if not db:
        print("No enrolled people found. Run 'enroll <name>' first.")
        return

    print(f"Loaded {len(db)} enrolled people: {', '.join(db.keys())}")

    cap = open_camera(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Running live verification. Press R to reset/scan again, Q to quit.")

    locked = False
    locked_name = None
    locked_score = None
    locked_at = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if locked:
            # Skip heavy inference while locked
            cv2.rectangle(display, (10, 10), (display.shape[1] - 10, 70), (0, 150, 0), -1)
            cv2.putText(
                display,
                f"VERIFIED: {locked_name} ({locked_score:.2f})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "Press R to scan again",
                (20, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            if AUTO_RESET_SECONDS is not None and (time.time() - locked_at) > AUTO_RESET_SECONDS:
                locked = False
                locked_name = None
                locked_score = None
                print("Auto-reset: scanning again.")

        else:
            emb, bbox = get_embedding(app, frame)

            if emb is not None:
                best_name, best_score = identify(emb, db)
                matched = best_score > THRESHOLD

                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = (0, 255, 0) if matched else (0, 0, 255)
                label = f"{best_name} ({best_score:.2f})" if matched else f"UNKNOWN ({best_score:.2f})"

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                if matched:
                    locked = True
                    locked_name = best_name
                    locked_score = best_score
                    locked_at = time.time()
                    print(f"Locked on: {best_name} ({best_score:.2f})")
            else:
                cv2.putText(
                    display,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Verification", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            locked = False
            locked_name = None
            locked_score = None
            print("Manual reset: scanning again.")

    cap.release()
    cv2.destroyAllWindows()


def list_people():
    db = load_database()
    if not db:
        print("No enrolled people found.")
        return
    print(f"{len(db)} enrolled people:")
    for name in db.keys():
        print(f"  - {name}")


def delete_person(name):
    db = load_database()
    if name not in db:
        print(f"'{name}' not found in database.")
        return
    del db[name]
    save_database(db)
    print(f"Removed '{name}'. {len(db)} people remain.")


def print_usage():
    print("Usage:")
    print("  python webcam_face_verify.py enroll <name>")
    print("  python webcam_face_verify.py verify")
    print("  python webcam_face_verify.py list")
    print("  python webcam_face_verify.py delete <name>")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("enroll", "verify", "list", "delete"):
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "enroll":
        if len(sys.argv) != 3:
            print("Error: 'enroll' requires a name, e.g. python webcam_face_verify.py enroll Willson")
            sys.exit(1)
        print("Loading model...")
        app = load_model()
        enroll(app, sys.argv[2])

    elif command == "verify":
        print("Loading model...")
        app = load_model()
        verify(app)

    elif command == "list":
        list_people()

    elif command == "delete":
        if len(sys.argv) != 3:
            print("Error: 'delete' requires a name, e.g. python webcam_face_verify.py delete Willson")
            sys.exit(1)
        delete_person(sys.argv[2])