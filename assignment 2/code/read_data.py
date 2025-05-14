import h5py
import os
import numpy as np

# TODO: use os to do relative path from the working directory

source_dir: str = "extracted_zip_in_here"

def get_dataset_name(file_name_with_dir, taskType) -> str:
    # TODO if no subject identifier is supplied return selection message
    # or option to obtain all (with warning)

    # TODO: obtain all files in source data directory
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('.')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

# TODO: Function that given a path reads in the data and return numpy array
def read_data(file_name_with_dir) -> np.ndarray:
    dataset_name = 
    with h5py.File(file_name_with_dir, 'r') as f:
        dataset_name = list(f.keys())[0]
        data = f[dataset_name][:]
