"""
Enhanced Dataset Creator for Hand Mudra/Gesture Recognition
============================================================
Python 3.11 compatible | Windows path support
Features:
  - Data augmentation (flip, rotate, scale, noise)
  - Progress bars via tqdm
  - Skips corrupt/unreadable images gracefully
  - Handles any image filename (no naming convention needed)
  - Saves one master CSV (all classes) + per-class CSVs
  - Logs skipped files for review

Folder structure expected:
  dataset/
  ├── Alapadma/
  │   ├── anything1.jpg
  │   ├── xyz_abc.png
  │   └── ...
  ├── Anjali/
  │   └── ...
  └── ...
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# ✅ Your dataset root — the folder that CONTAINS all mudra sub-folders
# Each sub-folder name = class/mudra label
IMAGE_ROOT = r"C:\My Space\naveen\Project_Program\nit_delhi\natyaAI\code\data\dataset"

# ✅ Output folder for CSVs — will be created next to this script
OUTPUT_DIR  = r"C:\My Space\naveen\Project_Program\nit_delhi\natyaAI\code\data\csv_output"
MASTER_CSV  = r"C:\My Space\naveen\Project_Program\nit_delhi\natyaAI\code\data\master_dataset.csv"
LOG_FILE    = r"C:\My Space\naveen\Project_Program\nit_delhi\natyaAI\code\data\skipped_images.log"

# ✅ Supported image extensions (covers any filename, any format)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Augmentation toggles
AUGMENT           = True
AUG_FLIP          = True      # horizontal flip
AUG_ROTATE        = True      # ±15 ° random rotation
AUG_SCALE         = True      # ±10 % random scale
AUG_NOISE         = True      # Gaussian landmark noise (σ=0.005)
AUG_BRIGHTNESS    = True      # random brightness ±40
AUG_COPIES        = 4         # extra augmented copies per image

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5
STATIC_IMAGE_MODE        = True
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(filename=LOG_FILE, level=logging.WARNING,
                    format="%(message)s", filemode="w")

mp_hands = mp.solutions.hands


def extract_landmarks(results) -> list[list[float]]:
    """Return a list of rows (one per detected hand), raw x,y per landmark."""
    rows = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            row = []
            for lm in hand.landmark:
                row.extend([lm.x, lm.y])
            rows.append(row)
    return rows


def augment_image(image: np.ndarray) -> list[np.ndarray]:
    """Return a list of augmented variants of the image."""
    variants = []
    h, w = image.shape[:2]

    for _ in range(AUG_COPIES):
        img = image.copy()

        if AUG_BRIGHTNESS:
            delta = np.random.randint(-40, 41)
            img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

        if AUG_FLIP and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        if AUG_ROTATE:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)

        if AUG_SCALE:
            scale = np.random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)

        variants.append(img)
    return variants


def add_landmark_noise(row: list[float], sigma: float = 0.005) -> list[float]:
    """Add small Gaussian noise to landmark coordinates."""
    arr = np.array(row)
    arr += np.random.normal(0, sigma, arr.shape)
    return arr.tolist()


def process_image(image: np.ndarray, hands_model) -> list[list[float]]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb)
    return extract_landmarks(results)


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    image_root = Path(IMAGE_ROOT)
    if not image_root.exists():
        print(f"[ERROR] Image root '{IMAGE_ROOT}' not found.")
        return

    class_dirs = [d for d in image_root.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"[ERROR] No class sub-folders found inside:\n  {IMAGE_ROOT}")
        print("  Make sure the path points to the folder CONTAINING mudra folders.")
        return

    print(f"Found {len(class_dirs)} class(es): {[d.name for d in class_dirs]}\n")

    all_rows = []   # (label, *landmark_coords)
    class_counts = {}

    with mp_hands.Hands(static_image_mode=STATIC_IMAGE_MODE,
                        max_num_hands=2,
                        min_detection_confidence=MIN_DETECTION_CONFIDENCE) as hands:

        for class_dir in tqdm(class_dirs, desc="Classes", unit="class"):
            label = class_dir.name
            img_files = [f for f in class_dir.iterdir()
                         if f.suffix.lower() in SUPPORTED_EXTS]

            class_rows = []

            for img_path in tqdm(img_files, desc=f"  {label}", leave=False, unit="img"):
                image = cv2.imread(str(img_path))
                if image is None:
                    logging.warning(f"SKIP (unreadable): {img_path}")
                    continue

                # ── Original image ──
                for row in process_image(image, hands):
                    class_rows.append(row)

                # ── Augmented copies ──
                if AUGMENT:
                    for aug_img in augment_image(image):
                        for row in process_image(aug_img, hands):
                            if AUG_NOISE:
                                row = add_landmark_noise(row)
                            class_rows.append(row)

            if not class_rows:
                print(f"  [WARN] No landmarks extracted for class '{label}' — skipping.")
                continue

            # Save per-class CSV
            df = pd.DataFrame(class_rows)
            out_path = Path(OUTPUT_DIR) / f"{label}.csv"
            df.to_csv(out_path, index=False)
            tqdm.write(f"  ✓ {label}: {len(class_rows)} samples → {out_path}")

            class_counts[label] = len(class_rows)

            # Accumulate for master CSV
            for row in class_rows:
                all_rows.append([label] + row)

    # ── Master CSV ──
    if all_rows:
        n_coords = len(all_rows[0]) - 1
        coord_cols = []
        for i in range(n_coords // 2):
            coord_cols += [f"x{i}", f"y{i}"]
        master_df = pd.DataFrame(all_rows, columns=["label"] + coord_cols)
        master_df.to_csv(MASTER_CSV, index=False)
        print(f"\n✅ Master CSV saved: '{MASTER_CSV}' ({len(all_rows)} total rows)")
    else:
        print("\n[ERROR] No data extracted at all. Check your images_dataset/ folder.")
        return

    print("\n── Class Distribution ──")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls:30s}: {cnt:>6} samples")
    print(f"\nExtraction complete! Check '{LOG_FILE}' for any skipped files.")


if __name__ == "__main__":
    main()