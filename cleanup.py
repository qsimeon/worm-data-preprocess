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
    parser = argparse.ArgumentParser(
        description="Clean up generated directories and files"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force deletion of all directories without confirmation",
    )
    parser.add_argument(
        "-c",
        "--connectome",
        action="store_true",
        help="Delete processed connectome files",
    )
    parser.add_argument(
        "-n", "--neural", action="store_true", help="Delete processed neural files"
    )
    parser.add_argument(
        "-d", "--downloaded", action="store_true", help="Delete downloaded files"
    )
    args = parser.parse_args()

    # Define directory groups
    connectome_dirs = ["data/processed/connectome"]
    neural_dirs = ["data/processed/neural"]
    downloaded_dirs = ["data/opensource_neural_data", "data/raw"]

    directories_to_remove = []

    # If -f flag is used, remove everything
    if args.force:
        directories_to_remove = connectome_dirs + neural_dirs + downloaded_dirs
    else:
        # Add directories based on flags
        if args.connectome:
            directories_to_remove.extend(connectome_dirs)
        if args.neural:
            directories_to_remove.extend(neural_dirs)
        if args.downloaded:
            directories_to_remove.extend(downloaded_dirs)

        # If no specific flags are set, prompt user for each option
        if not (args.connectome or args.neural or args.downloaded):
            print("No flags (-f, -n, -c, -d) specified:")

            if (
                input("Delete neural processed files? (yes/no): ").strip().lower()
                == "yes"
            ):
                directories_to_remove.extend(neural_dirs)

            if (
                input("Delete connectome processed files? (yes/no): ").strip().lower()
                == "yes"
            ):
                directories_to_remove.extend(connectome_dirs)

            if (
                input("Delete downloaded files (not recommended)? (yes/no): ").strip().lower()
                == "yes"
            ):
                directories_to_remove.extend(downloaded_dirs)

    if not directories_to_remove:
        print("No directories selected for removal. Exiting.")
        return

    print("The following directories will be deleted:")
    for directory in directories_to_remove:
        print(f" - {directory}")

    if (
        args.force
        or input("Are you sure you want to proceed? (yes/no): ").strip().lower()
        == "yes"
    ):
        remove_directories(directories_to_remove)
        print("Cleanup completed successfully.")
    else:
        print("Operation aborted.")


if __name__ == "__main__":
    main()
