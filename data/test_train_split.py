import os
import random

# Path to the 'image_2' folder
root = "/media/franco/hdd/dataset/dataset_3d/20250104_lvms_run02_virginia/camera_fr" 
image_folder = os.path.join(root, "image_2")

# Output text files
output_folder = os.path.join(root, "ImageSets")
os.makedirs(output_folder, exist_ok=True)
train_file = os.path.join(output_folder,"train.txt")
val_file = os.path.join(output_folder,"val.txt")
test_file = os.path.join(output_folder, "test.txt")
train_split = 0.8
val_split = 0.1

# Step 1: Get all filenames in the folder
all_files = [
    os.path.splitext(file)[0]  # Remove file extension
    for file in os.listdir(image_folder)
    if os.path.isfile(os.path.join(image_folder, file))
]

# Step 2: Shuffle and split the filenames
random.shuffle(all_files)
split_index = int(train_split * len(all_files))  # 80% split
split_index2 = split_index + int(val_split *len(all_files))

train_files = all_files[:split_index]
val_files = all_files[split_index:split_index2]
test_files = all_files[split_index2:]

# Step 3: Write the lists to text files
with open(train_file, "w") as train_f:
    train_f.write("\n".join(train_files))

with open(val_file, "w") as val_f:
    val_f.write("\n".join(val_files))

with open(test_file, "w") as test_f:
    test_f.write("\n".join(test_files))

print(f"Train and test files have been created:\n- {train_file}\n- {test_file}")
