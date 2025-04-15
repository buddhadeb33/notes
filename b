import os

# Path where all 'data_output...' folders exist
root_dir = "path/to/your/root/folder"  # ← Replace with actual root path

for folder_name in os.listdir(root_dir):
    old_folder_path = os.path.join(root_dir, folder_name)

    # Only process folders starting with 'data_output'
    if os.path.isdir(old_folder_path) and folder_name.startswith("data_output"):
        try:
            # Traverse into the subfolders to find the prefix under HAS/
            sub_path = os.path.join(
                old_folder_path,
                "model_training",
                "ALL MODEL 0312",
                "model_data",
                "HAS"
            )

            # Get the name like 'us_ach' or 'us_zba'
            subfolders = [
                f for f in os.listdir(sub_path)
                if os.path.isdir(os.path.join(sub_path, f))
            ]

            if not subfolders:
                print(f"❌ No prefix folder found inside HAS/ for: {folder_name}")
                continue

            prefix = subfolders[0]  # e.g., 'us_ach'
            new_folder_name = f"{prefix}_data"
            new_folder_path = os.path.join(root_dir, new_folder_name)

            # If folder with new name already exists, warn and skip to avoid overwrite
            if os.path.exists(new_folder_path):
                print(f"⚠️ Skipping: {new_folder_name} already exists.")
                continue

            os.rename(old_folder_path, new_folder_path)
            print(f"✅ Renamed: {folder_name} → {new_folder_name}")

        except Exception as e:
            print(f"⚠️ Error processing {folder_name}: {e}")
