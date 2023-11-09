#!/usr/bin/env python3
"""
Prepares a new dataset for nnUNetv2, focusing on myelin boundary segmentation.

Features:
- Mask Preparation: Generates a single mask per image. Marks the boundary
  between contacting myelin sheaths of different axons with a value of 1, 
  with 0 representing the background.
- Training Set Compilation: Includes all subjects with annotations in the 
  training set. nnUNetv2 will perform automatic cross-validation using these 
  annotated subjects.
- Testing Set Assignment: Allocates subjects without annotations to the 
  testing set, facilitating model performance evaluation on unseen data.
- Inspiration: The structure and methodology of this script is
  inspired by Armand Collin's work. The original script by Armand Collin can
  be found at:
  https://github.com/axondeepseg/model_seg_rabbit_axon-myelin_bf/blob/main/nnUNet_scripts/prepare_data.py
"""


__author__ = "Arthur Boschet"
__license__ = "MIT"


import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import cv2


def extract_sample_participant(s: str) -> tuple:
    """
    Extracts sample and participant identifiers from a given string and returns them as a tuple.

    Parameters
    ----------
    s : str
        The string from which identifiers are to be extracted.
        Expected to contain a pattern like 'sub-nyuMouseXX_sample-XXXX'.

    Returns
    -------
    tuple
        A tuple containing the participant and sample identifiers.
        For example, for "sub-nyuMouse26_sample-0002_axonmyelin.png", it returns ('sub-nyuMouse26', 'sample-0002').

    Raises
    ------
    ValueError
        If the string does not contain the expected pattern.
    """
    match = re.search(r"(sub-nyuMouse\d+)_.*(sample-\d+)", s)
    if not match:
        raise ValueError("The string does not contain the expected pattern.")
    return match.groups()


def create_directories(base_dir: str, subdirs: List[str]):
    """
    Creates subdirectories in a specified base directory.

    Parameters
    ----------
    base_dir : str
        The base directory where subdirectories will be created.
    subdirs : List[str]
        A list of subdirectory names to create within the base directory.
    """
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)


def save_json(data: Dict, file_path: str):
    """
    Saves a dictionary as a JSON file at the specified path.

    Parameters
    ----------
    data : Dict
        Dictionary to be saved as JSON.
    file_path : str
        File path where the JSON file will be saved.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def process_images(
    subject_list: List[str],
    datapath: Path,
    out_folder: str,
    case_id_dict: Dict[str, int],
    dataset_name: str,
    is_test: bool = False,
):
    """
    Processes all image files in each subject's directory.

    Parameters
    ----------
    subject_list : List[str]
        List of subjects whose images are to be processed.
    datapath : Path
        Path to the data directory.
    out_folder : str
        Output directory to save processed images.
    case_id_dict : Dict[str, int]
        Dictionary mapping subject names to case IDs.
    dataset_name : str
        Name of the dataset.
    is_test : bool, optional
        Boolean flag indicating if the images are for testing, by default False.
    """
    folder_type = "imagesTs" if is_test else "imagesTr"
    image_suffix = "_0000"
    for subject in subject_list:
        image_files = sorted(Path(datapath, subject, "micr").glob("*.png"))
        for img_file in image_files:
            key = str(extract_sample_participant(os.path.basename(img_file)))
            case_id = case_id_dict[key]
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            fname = f"{dataset_name}_{case_id:03d}{image_suffix}.png"
            cv2.imwrite(os.path.join(out_folder, folder_type, fname), img)


def process_labels(
    subject_list: List[str],
    datapath: Path,
    out_folder: str,
    case_id_dict: Dict[str, int],
    dataset_name: str,
):
    """
    Processes label images from a list of subjects, matching each image with the label having the largest 'N' number.

    Parameters
    ----------
    subject_list : List[str]
        List of subjects whose label images are to be processed.
    datapath : Path
        Path to the data directory.
    out_folder : str
        Output directory to save processed label images.
    case_id_dict : Dict[str, int]
        Dictionary mapping subject names to case IDs.
    dataset_name : str
        Name of the dataset.
    """
    for subject in subject_list:
        label_files = []
        image_files = sorted(
            Path(datapath, "derivatives", "labels", subject, "micr").glob("*.png")
        )
        for img_file in image_files:
            sample = os.path.basename(img_file).split("_")[1]
            label_files_match = sorted(
                Path(datapath, "derivatives", "labels", subject, "micr").glob(
                    f"{subject}_{sample}_axonmyelin_seg-touching*-manual.png"
                )
            )
            if label_files_match:
                label_files.append(label_files_match[-1])

        for label_file in label_files:
            key = str(extract_sample_participant(os.path.basename(label_file)))
            case_id = case_id_dict[key]
            label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE) // 255
            fname = f"{dataset_name}_{case_id:03d}.png"
            cv2.imwrite(os.path.join(out_folder, "labelsTr", fname), label)


def create_case_id_dict(subject_list: List[str], datapath: Path) -> Dict[str, int]:
    """
    Creates a dictionary mapping unique (sample_id, participant_id) tuples to case IDs.

    Parameters
    ----------
    subject_list : List[str]
        List of subjects whose images are to be processed.
    datapath : Path
        Path to the data directory.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping unique (sample_id, participant_id) tuples to case IDs.
    """
    case_id_dict = {}
    case_id = 0
    samples_file = datapath / "samples.tsv"
    with open(samples_file, "r") as f:
        # Skip header line
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            sample_id, participant_id = parts[0], parts[1]
            key = str((participant_id, sample_id))
            if participant_id in subject_list:
                if key not in case_id_dict:
                    case_id_dict[key] = case_id
                    case_id += 1
    return case_id_dict


def main(args):
    """
    Main function to process dataset for nnUNet.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing DATAPATH and TARGETDIR.
    """
    dataset_name = args.DATASETNAME
    description = args.DESCRIPTION
    datapath = Path(args.DATAPATH)
    target_dir = Path(args.TARGETDIR)
    derivatives = list(Path(datapath, "derivatives", "labels").glob("sub*"))
    subject_list = [d.name for d in derivatives]

    out_folder = os.path.join(target_dir, "nnUNet_raw", f"Dataset001_{dataset_name}")
    create_directories(out_folder, ["imagesTr", "labelsTr", "imagesTs"])

    case_id_dict = create_case_id_dict(subject_list, datapath)

    dataset_info = {
        "name": dataset_name,
        "description": description,
        "labels": {"background": 0, "boundary": 1},
        "channel_names": {"0": "rescale_to_0_1"},
        "numTraining": len(case_id_dict),
        "numTest": 0,
        "file_ending": ".png",
    }
    save_json(dataset_info, os.path.join(out_folder, "dataset.json"))

    process_images(subject_list, datapath, out_folder, case_id_dict, dataset_name)
    process_labels(subject_list, datapath, out_folder, case_id_dict, dataset_name)

    unannotated_subjects = [
        d.name for d in Path(datapath, "derivatives", "ads-derivatives").glob("sub*")
    ]
    process_images(
        unannotated_subjects,
        datapath,
        out_folder,
        case_id_dict,
        dataset_name,
        is_test=True,
    )

    save_json(case_id_dict, os.path.join(target_dir, "subject_to_case_identifier.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("DATAPATH", help="Path to the original dataset in BIDS format")
    parser.add_argument(
        "--TARGETDIR",
        default=".",
        help="Target directory for the new dataset, defaults to current directory",
    )
    parser.add_argument(
        "--DATASETNAME",
        default="MyelinBoundarySegmentation",
        help="Name of the new dataset, defaults to MyelinBoundarySegmentation",
    )
    parser.add_argument(
        "--DESCRIPTION",
        default="Myelin boundary segmentation dataset for nnUNetv2",
        help="Description of the new dataset, defaults to Myelin boundary segmentation dataset for nnUNetv2",
    )
    args = parser.parse_args()
    main(args)
