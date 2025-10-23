import os
import shutil
import json
import sys
import tarfile
import subprocess
import argparse

# ImageNet download URLs
TRAIN_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
VAL_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
VALPREP_URL = "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"

#----------------------------------------------------------------------------

def download_imagenet(target_path):
    print(f"--- Downloaing ImageNet in: {target_path} ---")

    # 1. Setup root directory
    os.makedirs(target_path, exist_ok=True)

    # 2. Download Archives
    print("\n--- 1. Downloading Archives ---")

    train_archive = os.path.join(target_path, 'ILSVRC2012_img_train.tar')
    val_archive = os.path.join(target_path, 'ILSVRC2012_img_val.tar')

    for url, path in [(TRAIN_URL, train_archive), (VAL_URL, val_archive)]:
        command = [
            'wget',
            url,
            '-O', path,
        ]

        print(f"Executing command: {' '.join(command)}")

        # Run the command
        subprocess.run(command, check=True)

        print(f"Successfully downloaded {url}")

def extract_imagenet(target_path):
    print("\n--- 2. Extracting Training Data ---")
    train_path = os.path.join(target_path, 'train')
    os.makedirs(train_path, exist_ok=True)
    train_archive = os.path.join(target_path, 'ILSVRC2012_img_train.tar')
    val_archive = os.path.join(target_path, 'ILSVRC2012_img_val.tar')
    print(f"Extracting {train_archive}...")

    with tarfile.open(train_archive, 'r') as tar:
        tar.extractall(path=train_path)
    os.remove(train_archive) # Delete the large archive

    print("Extracting class sub-archives...")
    # Navigate into the train directory to process nested archives
    for nested_archive in os.listdir(train_path):
        if nested_archive.endswith('.tar'):
            print(f"Extracting {nested_archive}...")
            class_name = nested_archive.split(".")[0] # e.g., 'n01440764'
            nested_archive_path = os.path.join(train_path, nested_archive)
            class_dir = os.path.join(train_path, class_name)
            os.makedirs(class_dir, exist_ok=True)

            with tarfile.open(nested_archive_path, 'r') as tar:
                tar.extractall(path=class_dir)
            os.remove(nested_archive_path)

    # 4. Process Validation Data
    print("\n--- 3. Extracting Validation Data ---")
    val_dir = os.path.join(target_path, 'val')
    os.makedirs(val_dir, exist_ok=True)

    print(f"Extracting {val_archive}...")

    with tarfile.open(val_archive, 'r') as tar:
        tar.extractall(path=val_dir)
    os.remove(val_archive) # Delete the archive

    print("Sorting validation images using valprep.sh logic...")
    # Download and run the valprep.sh script logic using Python's subprocess

    # 4a. Download valprep.sh
    valprep_script_path = os.path.join(val_dir, 'valprep.sh')
    command = [
        'wget',
        '-qO-', VALPREP_URL,
        '-O', valprep_script_path,
    ]

    subprocess.run(command, check=True)

    # Run the script from the 'val' directory as the shell script expects
    subprocess.run(['bash', 'valprep.sh'], cwd=val_dir, check=True)
    os.remove(valprep_script_path) # Clean up the script

    print("Validation data successfully organized into class folders.")

    print("\n--- ImageNet Extraction Complete! ---")

def reorganize_imagenet(input_root_dir):
    output_images_dir = input_root_dir
    output_meta_dir = input_root_dir
    # Track the number of operations performed
    print(f"\n--- 4. Reorganzing Data for {input_root_dir} ---")
    class_name_list = os.listdir(input_root_dir)
    class_name_list.sort()
    for class_id, item_name in enumerate(class_name_list):
        class_folder_path = os.path.join(input_root_dir, item_name)
        if os.path.isdir(class_folder_path):
            class_synset_id = item_name  # e.g., 'n01440764'
            create_meta(class_id, class_folder_path, output_images_dir, output_meta_dir)
            shutil.rmtree(class_folder_path)
    print("Process Complete.")

def create_meta(class_id, class_folder_path, output_images_dir, output_meta_dir):
    for file_name in os.listdir(class_folder_path):
        if file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.jpg'):
            original_file_path = os.path.join(class_folder_path, file_name)
            # Define paths for the new image and meta file
            new_image_path = os.path.join(output_images_dir, file_name.lower())
            # Create the base name for the meta JSON file (e.g., 'n01440764_10026.JPEG.meta.json')
            meta_file_name = file_name.split(".")[0].lower() + '.meta.json'
            meta_file_path = os.path.join(output_meta_dir, meta_file_name)
            shutil.move(original_file_path, new_image_path)
            meta_data = {
                'class': class_id  # Store the Synset ID as the class label
            }
            with open(meta_file_path, 'w') as f:
                json.dump(meta_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(
        description="Download, extract, and prepare ImageNet dataset."
    )
    parser.add_argument(
        "--target_directory", # ADD the leading double-dash
        type=str,
        help="Path to the directory where ImageNet will be downloaded and processed."
    )

    args = parser.parse_args()

    download_imagenet(args.target_directory)
    extract_imagenet(args.target_directory)
    reorganize_imagenet(os.path.join(args.target_directory, 'train'))
    reorganize_imagenet(os.path.join(args.target_directory, 'val'))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
