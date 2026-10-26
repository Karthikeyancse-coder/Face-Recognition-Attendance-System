
import cv2
from pathlib import Path

# Input: images collected in Step 1
INPUT_DIR = Path("dataset")

# Output: prepared images for CNN training
OUTPUT_DIR = Path("processed_dataset")

IMAGE_SIZE = (128, 128)

# Staff and student folders
categories = ["staff", "students"]

total_processed = 0
total_skipped = 0

for category in categories:
    category_dir = INPUT_DIR / category

    if not category_dir.exists():
        print(f"Folder not found: {category_dir}")
        continue

    # Find each person's folder
    for person_dir in sorted(category_dir.iterdir()):

        if not person_dir.is_dir():
            continue

        output_person_dir = (
            OUTPUT_DIR / category / person_dir.name
        )
        output_person_dir.mkdir(
            parents=True, exist_ok=True
        )

        print(f"\nProcessing: {person_dir.name}")

        # Read every JPG, JPEG or PNG image
        image_files = [
            p for p in person_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        for image_path in sorted(image_files):

            # Load image
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipped unreadable image: {image_path.name}")
                total_skipped += 1
                continue

            # The Step 1 program already cropped the face.
            # Resize to ensure every image has the same dimensions.
            face = cv2.resize(
                image,
                IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )

            # Save using the same filename and person folder
            output_path = output_person_dir / (
                image_path.stem + ".jpg"
            )

            saved = cv2.imwrite(str(output_path), face)

            if saved:
                total_processed += 1
            else:
                print(f"Could not save: {output_path}")
                total_skipped += 1

        print(f"Images found: {len(image_files)}")

print("\n===== PREPROCESSING COMPLETE =====")
print(f"Processed images: {total_processed}")
print(f"Skipped images: {total_skipped}")
print(f"Output folder: {OUTPUT_DIR.resolve()}")
