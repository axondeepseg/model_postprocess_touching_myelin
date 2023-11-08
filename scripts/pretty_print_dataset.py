import argparse
import os
from typing import Optional


def print_directory_structure(rootdir: str, prefix: Optional[str] = "") -> None:
    """
    Recursively prints the directory structure with a pretty format.

    Parameters
    ----------
    rootdir : str
        The root directory from which to print the structure.
    prefix : str, optional
        The prefix used for formatting the directory structure, by default "".

    Returns
    -------
    None
    """
    files = os.listdir(rootdir)
    for index, file in enumerate(sorted(files)):
        path = os.path.join(rootdir, file)
        if os.path.isdir(path):
            # Directory case: recursively print its contents with an updated prefix
            new_prefix = prefix + "│   " if index < len(files) - 1 else prefix + "    "
            connector = "├── " if index < len(files) - 1 else "└── "
            print(f"{prefix}{connector}{file}")
            print_directory_structure(path, new_prefix)
        else:
            # File case: just print the file name
            connector = "├── " if index < len(files) - 1 else "└── "
            print(f"{prefix}{connector}{file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("DIRPATH", type=str, help="Path to the directory to print")
    args = parser.parse_args()

    print_directory_structure(args.DIRPATH)
