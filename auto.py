import os
import random
import shutil

dataset_path = "dataset"
output_path = "split_dataset"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

classes = os.listdir(dataset_path)

for cls in classes:

    class_path = os.path.join(dataset_path, cls)
    images = os.listdir(class_path)

    random.shuffle(images)

    train_end = int(len(images) * train_ratio)
    val_end = int(len(images) * (train_ratio + val_ratio))

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split_name, split_images in zip(
        ["train","validation","test"],
        [train_images, val_images, test_images]
    ):

        split_dir = os.path.join(output_path, split_name, cls)
        os.makedirs(split_dir, exist_ok=True)

        for img in split_images:

            src = os.path.join(class_path, img)
            dst = os.path.join(split_dir, img)

            shutil.copy(src, dst)

print("Dataset successfully split.")