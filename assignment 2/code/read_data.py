{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reading of the data\n",
    "The files contained in each of those folders have the “h5” extension. In order\n",
    "to read them, you need to use the h5py library ( that you can install using\n",
    "“pip install h5py” if you don’t have it already ). This type of files can contain\n",
    "datasets identified by a name. For simplicity, each file contains only 1 dataset.\n",
    "The following code snippet can read the file ”Intra/train/rest 105923 1.h5”:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import h5py\n",
    "import os\n",
    "\n",
    "# TODO: code that will obtain the shared directory will collectively using OS libary\n",
    "shared_directory = \"\"\n",
    "\n",
    "def get_dataset_name(file_name_with_dir):\n",
    "    filename_without_dir = file_name_with_dir.split('/')[-1]\n",
    "    temp = filename_without_dir.split('.')[:-1]\n",
    "    dataset_name = \"_\".join(temp)\n",
    "    return dataset_name\n",
    "\n",
    "filename_path = \"Intra/train/rest_105923_1.h5\"\n",
    "\n",
    "# Reads in data based on some file name path as above\n",
    "with h5py.File(filename_path, 'r') as f:\n",
    "    dataset_name = get_dataset_name(filename_path)\n",
    "    matrix = f.get(dataset_name)[()]\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
