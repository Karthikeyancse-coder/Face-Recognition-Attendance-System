# mark_attendance.py
import csv
from pathlib import Path
from datetime import datetime

ATTENDANCE_DIR = Path(__file__).resolve().parent / "attendance_records"
ATTENDANCE_DIR.mkdir(exist_ok=True)

# Person ID -> Real Name mapping
PERSON_NAMES = {
    "ST001": "SUJITH KUMAR M",
    "SD001": "DEV MK",
    "SD002": "SIVARAMAN P",
    "SD003": "SRINIVASAN M",
    "SD004": "MANIGANDA BALAJI D",
    "SD005": "NITHISH J",
    "SD006": "SUDHISH VIJAI",
    "SD007": "MEGHA JP",
    "SD008": "THAMARAI LAKSHMI S",
    "SD009": "PRAGATEESH V",
    "SD010": "SARVESH P",   # <-- Fixed: was "SD0010"
}


def mark_attendance(person_id):
    """Mark attendance for a person. Returns True if newly marked, False if duplicate."""
    now = datetime.now()
    file_path = ATTENDANCE_DIR / f"attendance_{now:%Y-%m-%d}.csv"

    person_name = PERSON_NAMES.get(person_id, "Name Not Added")
    file_exists = file_path.exists()

    # Prevent duplicate attendance for the same person on the same day.
    if file_exists:
        with open(file_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("Person ID") == person_id:
                    print(f"{person_name} ({person_id}): Already marked present today.")
                    return False

    with open(file_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Person ID", "Name", "Date", "Time", "Status"])

        writer.writerow([
            person_id,
            person_name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            "Present"
        ])

    print(f"Attendance marked: {person_name} ({person_id}) at {now:%H:%M:%S}")
    return True


def get_person_name(person_id):
    """Helper to get name from ID."""
    return PERSON_NAMES.get(person_id, "Name Not Added")