"""
Undoes the result of preprocess.py by deleting all generates directories and
files.
"""

import os
import shutil
import argparse

def remove_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Removed directory: {directory}")
        else:
            print(f"Directory does not exist: {directory}")

def main():
    parser = argparse.ArgumentParser(description='Clean up generated directories and files')
    parser.add_argument('-f', '--force', action='store_true', help='Force deletion without confirmation')
    args = parser.parse_args()

    directories_to_remove = [
        # 'data/opensource_neural_data', # this takes so long to download
        'data/processed',
        'data/raw'
    ]

    if not args.force:
        print("WARNING: This script will permanently delete the following directories and all their contents:")
        for directory in directories_to_remove:
            print(f" - {directory}")

    if args.force or input("Are you sure you want to proceed? (yes/no): ").strip().lower() == 'yes':
        remove_directories(directories_to_remove)
        print("Directories removed successfully.")
    else:
        print("Operation aborted.")

if __name__ == "__main__":
    main()