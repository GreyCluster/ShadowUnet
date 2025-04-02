import os
import random
import shutil
import numpy as np

data_path = r"D:\Projects\AI\Data\NeuralSM\Train\shadow"
images_path = os.path.join(data_path, "images")
condition_path = os.path.join(data_path, "conditioning_images")

train_images_path = os.path.join(images_path, "train")
val_images_path = os.path.join(images_path, "val")


if __name__ == '__main__':
    images_list = []
    for root, dirs, files in os.walk(images_path):
        for file in files:
            images_list.append(file)

    random.shuffle(images_list)
    split_index = int(len(images_list) * 0.9)
    train_list = images_list[:split_index]
    val_list = images_list[split_index:]

    # train
    for fname in train_list:
        old_file_path = os.path.join(images_path, fname)
        dst_file_path = os.path.join(train_images_path, fname)
        shutil.copyfile(old_file_path, dst_file_path)

    for fname in val_list:
        old_file_path = os.path.join(images_path, fname)
        dst_file_path = os.path.join(val_images_path, fname)
        shutil.copyfile(old_file_path, dst_file_path)