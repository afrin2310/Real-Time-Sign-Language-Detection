"""
extract_landmarks.py
────────────────────
Reads TRAIN dataset images and extracts MediaPipe hand landmarks.
Saves one landmarks.npy per label folder.

Usage:
    python extract_landmarks.py
"""

import os
import cv2
import numpy as np
from hand_detection import extract_landmarks

# Dataset paths
TRAIN_PATH = "dataset/asl_alphabet_train"
TEST_PATH = "dataset/asl_alphabet_test"

# Supported image formats
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_dataset():

    # Get all label folders from TRAIN folder
    labels = sorted([
        d for d in os.listdir(TRAIN_PATH)
        if os.path.isdir(os.path.join(TRAIN_PATH, d))
    ])

    # Check if labels exist
    if not labels:
        print(f"❌ No label folders found in '{TRAIN_PATH}'")
        return

    print(f"\n✅ Found {len(labels)} labels\n")

    total_saved = 0
    total_failed = 0

    # Process each label folder
    for label in labels:

        label_dir = os.path.join(TRAIN_PATH, label)

        # Output file
        out_path = os.path.join(label_dir, "landmarks.npy")

        # Read all images
        images = [
            f for f in os.listdir(label_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ]

        if not images:
            print(f"⚠️ {label}: No images found")
            continue

        print(f"\n📂 Processing '{label}' → {len(images)} images")

        vectors = []
        failed = 0

        # Process images
        for img_name in images:

            img_path = os.path.join(label_dir, img_name)

            frame = cv2.imread(img_path)

            if frame is None:
                failed += 1
                continue

            # Extract landmarks
            features, _, _ = extract_landmarks(frame, static_mode=True)

            # Save only valid hand detections
            if np.any(features != 0):
                vectors.append(features)
            else:
                failed += 1

        # Save landmarks
        if vectors:

            arr = np.array(vectors, dtype=np.float32)

            np.save(out_path, arr)

            print(f"✅ Saved {len(vectors)} samples")
            print(f"❌ Skipped {failed} images")

            total_saved += len(vectors)
            total_failed += failed

        else:
            print(f"❌ No hands detected in '{label}'")

    print("\n=================================")
    print(f"✅ TOTAL SAVED   : {total_saved}")
    print(f"❌ TOTAL SKIPPED : {total_failed}")
    print("=================================")

    print("\n🚀 Now run:")
    print("python train.py")


if __name__ == "__main__":
    process_dataset()