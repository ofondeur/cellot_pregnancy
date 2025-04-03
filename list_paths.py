import os
import pandas as pd
import anndata as ad
import re
import numpy as np


def create_list_of_paths2(
    directory, stimulation, cell_type=None, sample=None, patient_excluded=[]
):
    paths_list = []
    for filename in os.listdir(directory):
        if len(patient_excluded) > 0:
            for patient in patient_excluded:
                if (
                    stimulation in filename
                    and (cell_type is None or cell_type in filename)
                    and (sample is None or sample in filename)
                    and patient not in filename
                ):
                    paths_list.append([os.path.join(directory, filename),cell_type])
        else:
            if (
                stimulation in filename
                and (cell_type is None or cell_type in filename)
                and (sample is None or sample in filename)
            ):
                paths_list.append([os.path.join(directory, filename),cell_type])
    return paths_list


def create_list_of_paths_spec_patients(
    directory, stimulation, cell_type=None, sample=None, patient=None
):
    paths_list = []
    for filename in os.listdir(directory):
        if (
            stimulation in filename
            and (cell_type is None or cell_type in filename)
            and (sample is None or sample in filename)
            and (patient is None or patient in filename)
        ):
            paths_list.append([os.path.join(directory, filename),cell_type])
    return paths_list


def count_files(directory, stimulation):
    count = 0
    sumsize = 0
    for filename in os.listdir(directory):
        if filename.endswith(".fcs") and stimulation in filename:
            count += 1
            sumsize += os.path.getsize(os.path.join(directory, filename))
    return count, sumsize
