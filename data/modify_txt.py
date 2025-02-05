import os

# Path to the folder containing the files
# folder_path = "/media/franco/hdd/dataset/dataset_3d/20250104_lvms_run02_virginia/camera_fc/calib"
folder_path = "/media/franco/hdd/dataset/dataset_3d/20250104_lvms_run02_virginia/camera_fc/label_2"  # Replace with the path to your folder

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Process only .txt files
    if os.path.isfile(file_path) and filename.endswith(".txt"):
        # Open the file, read its contents, and replace the target word
        with open(file_path, "r") as file:
            content = file.read()
        
        # Replace the target word
        updated_content = content.replace("car", "Car")
        # updated_content = updated_content.replace("T_r_imu_to_velo", "Tr_imu_to_velo")
        # updated_content = updated_content.replace("T_r_velo_to_cam", "Tr_velo_to_cam")

        
        # Write the updated content back to the file
        with open(file_path, "w") as file:
            file.write(updated_content)
        
        print(f"Updated content in: {filename}")

print("All files have been updated.")