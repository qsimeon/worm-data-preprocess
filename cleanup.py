"""
Undoes the result of preprocess.py by deleting all generates directories and
files.
"""

import os
import shutil

def remove_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Removed directory: {directory}")
        else:
            print(f"Directory does not exist: {directory}")

def main():
    directories_to_remove = [
        'data/opensource_neural_data',
        'data/processed',
        'data/raw'
    ]

    print("WARNING: This script will permanently delete the following directories and all their contents:")
    for directory in directories_to_remove:
        print(f" - {directory}")

    confirm = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
    if confirm == 'yes':
        remove_directories(directories_to_remove)
        print("Directories removed successfully.")
    else:
        print("Operation aborted.")

if __name__ == "__main__":
    main()