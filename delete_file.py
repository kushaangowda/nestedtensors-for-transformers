import os
import sys
import shutil


DIRS_TO_IGNORE = [".git"]
FILES_TO_IGNORE = [".gitignore", "README.md", "LICENSE"]


# Check contents with details
def delete_git_folder(directory_path):
    print(f"Contents of '{directory_path}':")
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        if os.path.isfile(item_path):
            if item not in FILES_TO_IGNORE:
                print(f"Removing file: {item}")
                os.remove(item_path)

        elif os.path.isdir(item_path):
            if item not in DIRS_TO_IGNORE:
                print(f"Removing directory: {item}")
                shutil.rmtree(item_path)

        else:
            print(f"Other: {item}")

if __name__ == "__main__":
    # Specify the directory path
    directory_path = sys.argv[1]
    delete_git_folder(directory_path)