import os
import pandas as pd


def non_overlapping_moving_average_repeated(data, M):
    # Reshape data into chunks of size M (the last chunk might be smaller if data size is not divisible by M)
    chunks = np.array_split(data, len(data) // M)

    # Compute average for each chunk and repeat the average M times (or the length of the chunk for the last segment)
    avg_values_repeated = np.concatenate([np.full(len(chunk), np.mean(chunk)) for chunk in chunks])

    return avg_values_repeated
def append_dict_to_excel(filename, data_dict):
    """Append a dictionary to an Excel file.

    Args:
        filename (str): The path to the Excel file.
        data_dict (dict): The dictionary to be appended as rows in the Excel file.

    Raises:
        IOError: If the file cannot be accessed.

    Notes:
        - If the file already exists, the dictionary data will be appended without writing the headers.
        - If the file doesn't exist, a new file will be created with the dictionary data and headers.

    """
    df = pd.DataFrame(data_dict)

    if os.path.isfile(filename):
        # If it exists, append without writing the headers
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # If it doesn't exist, create a new one with headers
        df.to_csv(filename, mode='w', header=True, index=False)


def print_directory_structure(startpath):
    """Print the directory structure recursively starting from a given path.

    Args:
        startpath (str): The path of the directory to print the structure from.

    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


import re
from fractions import Fraction
from scipy.signal import resample_poly
import numpy as np


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)


def resample(x, sr1, sr2, axis=0):
    '''sr1: target, sr2: source'''
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis).astype(np.float32)


def smooth_signal(y, n):
    box = np.ones(n) / n
    ys = np.convolve(y, box, mode='same')
    return ys


def zscore(x):
    return (x - np.mean(x, 0, keepdims=True)) / np.std(x, 0, keepdims=True)
