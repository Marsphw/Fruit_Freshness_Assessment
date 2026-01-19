import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

# ======================================================
#                Basic Path Configuration
# ======================================================
input_dir = "../data/raw"                 # Raw data (multi-level directory)
output_dir = "../data/processed"      # Augmented output directory
csv_path = "../dataset.csv"           # Output CSV file
splits_dir = "splits"              # Directory for train/val/test splits

os.makedirs(output_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)

# ======================================================
#          Counter for each key (fruit_condition)
# ======================================================
counter = {}
rows = []

# ======================================================
#             Augmentation: Salt and Pepper Noise
# ======================================================
def add_salt_pepper_noise(img, amount=0.01):
    img_np = np.array(img)
    h, w, c = img_np.shape
    num_noise = int(amount * h * w)

    # Salt (White pixels)
    for _ in range(num_noise):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        img_np[y, x] = [255, 255, 255]

    # Pepper (Black pixels)
    for _ in range(num_noise):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        img_np[y, x] = [0, 0, 0]

    return Image.fromarray(img_np)

# ======================================================
#   Generate 3 augmentations per image (Random combination)
# ======================================================
def generate_augmentations(img, num_aug=3):
    results = []
    for _ in range(num_aug):
        aug = img.copy()

        # Random Flip
        if random.random() < 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)

        # Random Rotation
        if random.random() < 0.5:
            angle = random.randint(-25, 25)
            aug = aug.rotate(angle)

        # Random Brightness
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(aug)
            aug = enhancer.enhance(random.uniform(0.7, 1.4))

        # Salt and Pepper Noise
        if random.random() < 0.5:
            aug = add_salt_pepper_noise(aug, amount=0.01)

        results.append(aug)

    return results

# ======================================================
#       Traverse Multi-level Directory & Process Images
# ======================================================
for fruit in os.listdir(input_dir):
    fruit_path = os.path.join(input_dir, fruit)
    if not os.path.isdir(fruit_path):
        continue

    for condition in os.listdir(fruit_path):
        cond_path = os.path.join(fruit_path, condition)
        if not os.path.isdir(cond_path):
            continue

        fruit_lower = fruit.lower()
        cond_lower = condition.lower()
        key = f"{fruit_lower}_{cond_lower}"

        counter.setdefault(key, 0)

        out_subdir = os.path.join(output_dir, fruit, condition)
        os.makedirs(out_subdir, exist_ok=True)

        # Traverse images in subdirectory
        for img_name in os.listdir(cond_path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(cond_path, img_name)
            img = Image.open(img_path).convert("RGB").resize((224, 224))

            # ---------------- Save Original Image ----------------
            counter[key] += 1
            new_name = f"{key}_{counter[key]:04d}.jpg"
            save_path = os.path.join(out_subdir, new_name)
            img.save(save_path)

            rel_path = os.path.relpath(save_path, output_dir)
            rows.append([rel_path, fruit_lower, cond_lower])

            # ---------------- Generate 3 Augmented Images ----------------
            aug_images = generate_augmentations(img, num_aug=3)

            for aug in aug_images:
                counter[key] += 1
                new_name = f"{key}_{counter[key]:04d}.jpg"
                save_path = os.path.join(out_subdir, new_name)
                aug.save(save_path)

                rel_path = os.path.relpath(save_path, output_dir)
                rows.append([rel_path, fruit_lower, cond_lower])

# ======================================================
#                     Export to CSV
# ======================================================
df = pd.DataFrame(rows, columns=["path", "fruit", "condition"])
df.to_csv(csv_path, index=False)
print("CSV export completed:", csv_path)