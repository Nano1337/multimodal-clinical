import os

def check_empty_subdirectories(directory):
    """
    This function checks for empty subdirectories within a given directory.
    It prints the paths of empty subdirectories if found.
    """
    empty_dirs = []
    for root, dirs, files in os.walk(directory):
        if not dirs and not files:  # If there are no subdirectories and no files
            empty_dirs.append(root)
    
    if empty_dirs:
        print("Empty subdirectories found:")
        for dir in empty_dirs:
            print(dir)
    else:
        print("No empty subdirectories found.")

if __name__ == "__main__":
    directory_to_check = "../data/vggsound/train_Image-01-FPS"
    check_empty_subdirectories(directory_to_check)
