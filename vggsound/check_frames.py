import os

def check_subdirectories_with_min_files(directory, min_file_count=1):
    """
    This function checks for subdirectories within a given directory that have less than or equal to a specified minimum number of files.
    It prints the paths of these subdirectories if found.
    """
    dirs_with_min_files = []
    for root, dirs, files in os.walk(directory):
        if len(files) <= min_file_count:
            dirs_with_min_files.append((root, len(files)))
    
    if dirs_with_min_files:
        print(f"Subdirectories with {min_file_count} or fewer files found:")
        for dir, count in dirs_with_min_files:
            print(f"{dir} - {count} files")
    else:
        print(f"No subdirectories with {min_file_count} or fewer files found.")

if __name__ == "__main__":
    directory_to_check = "../data/vggsound/test_Image-01-FPS"
    check_subdirectories_with_min_files(directory_to_check, 6)
