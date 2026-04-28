import os
import shutil
import random

# Paths
SOURCE_DIR = "data/raw/chest_data"
DEST_DIR = "data/processed"

# Split ratio
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Create folders
for split in ["train", "val", "test"]:
    for class_name in os.listdir(SOURCE_DIR):
        os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)

# Split data
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)
    train_end = int(TRAIN_RATIO * total)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * total)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(DEST_DIR, "train", class_name, img))

    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(DEST_DIR, "val", class_name, img))

    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(DEST_DIR, "test", class_name, img))

print("✅ Data split completed!")