import h5py
import os
import numpy as np
from typing import List, Optional, Union

# This script is located in assignment2/code/read_data.py
# The folder "extracted_zip_in_here" is located in assignment2/extracted_zip_in_here
# and contains "Final Project data/Cross" and "Final Project data/Intra"
source_dir: str = os.path.join("..", "extracted_zip_in_here", "Final Project data")

# Nomenclature used by the provided data
VALID_TYPE_DATA = ["Cross", "Intra"]
VALID_TASK_TYPES = ["rest", "task_motor", "task_story_math", "task_working_memo"]

def get_dataset_name(file_name: Optional[str] = None, taskType: Optional[str] = None,
                     typeData: Optional[str] = None) -> Union[List[str], str, None]:
    
    # If no type of data is provided, list the available data types
    if typeData is None:
        print("Available typeData options:", VALID_TYPE_DATA)
        return None
    
    if typeData not in VALID_TYPE_DATA:
        print(f"Invalid typeData '{typeData}'. Choose from: {VALID_TYPE_DATA}")
        return None

    # Grow the folder path based on the type of data selected
    data_folder = os.path.join(source_dir, typeData, "train")

    # If no taskType is provided, list the available task types
    if taskType is None:
        print("Available taskType options:", VALID_TASK_TYPES)
        return None

    if taskType not in VALID_TASK_TYPES:
        print(f"Invalid taskType '{taskType}'. Choose from: {VALID_TASK_TYPES}")
        return None

    # If no file_name is provided, list the available subject identifiers you can choose from then
    # list subject identifiers
    if file_name is None:
        identifiers = set()
        for fname in os.listdir(data_folder):
            if fname.startswith(taskType):
                parts = fname.split('_')
                if len(parts) >= 3:
                    identifiers.add(parts[1])
        identifiers = sorted(list(identifiers))
        print("Available subject identifiers:", identifiers + ["all"])
        return None
    
    matched_files = []
    
    if file_name == "all":
        for fname in os.listdir(data_folder):
            if fname.startswith(taskType) and fname.endswith(".h5"):
                matched_files.append(os.path.join(data_folder, fname))
        return matched_files if matched_files else None

    for fname in os.listdir(data_folder):
        if fname.startswith(f"{taskType}_{file_name}_") and fname.endswith(".h5"):
            matched_files.append(os.path.join(data_folder, fname))
    
    if not matched_files:
        print(f"No files found for subject '{file_name}' under taskType '{taskType}' and typeData '{typeData}'")
        return None

    return matched_files

def read_data(file_names_with_dir: Union[str, List[str]]) -> dict:
    if isinstance(file_names_with_dir, str):
        file_names_with_dir = [file_names_with_dir]

    data_by_subject = {}
    
    for path in file_names_with_dir:
        file_name = os.path.basename(path)
        # Parse subject identifier
        parts = file_name.split('_')
        if len(parts) < 3:
            continue  # Unexpected filename format
        subject_id = parts[1]

        with h5py.File(path, 'r') as f:
            dataset_name = list(f.keys())[0]
            data = f[dataset_name][:]
            if subject_id not in data_by_subject:
                data_by_subject[subject_id] = []
            data_by_subject[subject_id].append(data)

    return data_by_subject
