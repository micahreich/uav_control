import os


def find_package_root(current_file):
    # Get the absolute path of the current file
    current_path = os.path.realpath(current_file)

    while current_path != os.path.dirname(current_path):  # Stop when reaching the root directory
        current_path = os.path.dirname(current_path)
        setup_path = os.path.join(current_path, ".git")
        if os.path.isdir(setup_path):
            return current_path  # Return the directory containing setup.py

    return None  # Return None if setup.py wasn't found
