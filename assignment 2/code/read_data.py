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

def get_dataset_name_train(file_name: Optional[str] = None, taskType: Optional[str] = None,
                     typeData: Optional[str] = None) -> Union[List[str], str, None]:
    
    # If no type of data is provided, list the available data types
    if typeData is None:
        print("Available typeData options:", VALID_TYPE_DATA)
        return None
    
    if typeData not in VALID_TYPE_DATA:
        print(f"Invalid typeData '{typeData}'. Choose from: {VALID_TYPE_DATA}")
        return None

    # Grow the folder path based on the type of data selected
    print("Loading train data")
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

def load_split_files(file_names_with_dir: Union[str, List[str]]) -> dict:
    if isinstance(file_names_with_dir, str):
        file_names_with_dir = [file_names_with_dir]

    data_by_file = {}
    
    for path in file_names_with_dir:
        file_name = os.path.basename(path)
        name_without_ext = os.path.splitext(file_name)[0]

        with h5py.File(path, 'r') as f:
            dataset_name = list(f.keys())[0]
            data = f[dataset_name][:]
            data_by_file[name_without_ext] = data

    return data_by_file

# # Load HDF5 file
# file_path = 'your_file_path_here.h5'
# with h5py.File(file_path, 'r') as f:
#     print("Top-level keys:", list(f.keys()))
    
#     # Example: explore one subject
#     subject_id = '113922'
#     if subject_id in f:
#         subject_data = f[subject_id]
#         print(f"Keys for subject {subject_id}:", list(subject_data.keys()))
        
#         # Check attributes that may contain sampling rate or metadata
#         if hasattr(subject_data, 'attrs'):
#             for key in subject_data.attrs:
#                 print(f"{key} : {subject_data.attrs[key]}")
