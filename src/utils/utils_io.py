import os
from collections import defaultdict
from typing import List
import glob

from scipy.io import readsav
import numpy as np

def print_sav_contents(filename, max_preview=10):
    """
    Legge un file .sav IDL e stampa tutte le chiavi, tipi e preview dei dati.

    Parameters
    ----------
    filename : str
        Percorso al file .sav.
    max_preview : int
        Numero massimo di elementi da stampare per preview array (default 10).
    """
    print(f"\n Reading SAV file: {filename}")
    data = readsav(filename, python_dict=True)
    print(type(data))
    print(f"\n Found {len(data)} keys:\n{'-'*50}")
    for key in data:
        val = data[key]
        print(f"\n Key: '{key}'")
        print(f"   Type: {type(val)}")
        if isinstance(val, np.ndarray):
            print(f"   Shape: {val.shape}")
            if val.dtype.names:
                print("   Structure fields:", val.dtype.names)
                for i, record in enumerate(val):
                    print(f"\n  Record {i}:")
                    for field in val.dtype.names:
                        print(f"      - {field}: {record[field]}")
                    if i >= max_preview - 1:
                        print("      ... (truncated)")
                        break
            else:
                print("   Preview:", val.ravel()[:max_preview])
        else:
            print(f"   Value:", val)

    print("\n Done.")
    
    
    
def find_files_in_directory(directory: str, search_string: str) -> List[str]:
    """
    Finds and returns a sorted list of file paths in a directory 
    that contain a specific string in their filename.

    Parameters
    ----------
    directory : str
        The path to the directory to search (e.g., '/path/to/data').
    search_string : str
        The string pattern to search for within the filenames 
        (e.g., 'NB_Dark', 'image', or '.txt').

    Returns
    -------
    List[str]
        A sorted list of full file paths matching the criteria.

    Examples
    --------
    >>> # Assuming 'data/' contains 'NB_Dark_001.fits' and 'NB_Dark_002.fits'
    >>> # find_files_in_directory('./data', 'NB_Dark')
    >>> # ['data/NB_Dark_001.fits', 'data/NB_Dark_002.fits']
    """
    
    # 1. Construct the pattern using os.path.join for cross-platform compatibility
    # The '*' wildcard before and after the search string ensures it matches 
    # any file where the string appears anywhere in the name.
    # E.g., 'path/to/directory/*NB_Dark*.fits' (if you were specifically looking for .fits)
    # E.g., 'path/to/directory/*NB_Dark*' (to search any file extension)
    search_pattern = os.path.join(directory, f'*{search_string}*')
    
    # 2. Use glob.glob to find all files matching the pattern
    # glob.glob returns a list of paths
    all_matching_files = glob.glob(search_pattern)
    
    # 3. Sort the list (as done in the inspiration example) and return
    # Sorting ensures a consistent, predictable order (e.g., numerical order for files)
    return sorted(all_matching_files)



def get_wavelengths(files: List[str]) -> List[int]:
    """
    Extract the unique wavelengths from a list of FITS file paths.

    This function assumes that the wavelength is encoded in the filename,
    as the first element before an underscore (``_``), e.g. ``656_file1.fits``.
    Only numeric values are considered valid wavelengths.

    Parameters
    ----------
    files : list of str
        List of file paths.

    Returns
    -------
    list of int
        Sorted list of unique wavelengths found in the filenames.

    Examples
    --------
    >>> files = ["656_image1.fits", "486_image2.fits", "656_image3.fits"]
    >>> get_wavelengths(files)
    [486, 656]
    """
    wavelengths = set()
    for f in files:
        wl = os.path.basename(f).split('_')[0]
        if wl.isdigit():
            wavelengths.add(int(wl))
    return sorted(wavelengths)

def get_files_for_wavelength(files: List[str], wavelength: int) -> List[str]:
    """
    Return all FITS files corresponding to a given wavelength.

    Parameters
    ----------
    files : list of str
        List of file paths.
    wavelength : int
        Wavelength to filter by.

    Returns
    -------
    list of str
        List of files associated with the given wavelength.

    Examples
    --------
    >>> files = ["656_image1.fits", "486_image2.fits", "656_image3.fits"]
    >>> get_files_for_wavelength(files, 656)
    ["656_image1.fits", "656_image3.fits"]
    """
    wl_files = []
    for f in files:
        wl = os.path.basename(f).split('_')[0]
        if wl.isdigit() and int(wl) == wavelength:
            wl_files.append(f)
    return wl_files


import shutil
import os
from typing import Optional

def copy_directory_contents(src_folder: str, dst_folder: str, exist_ok: bool = True) -> Optional[str]:
    """
    Recursively copies a source directory and all its contents to a destination directory.

    If the destination directory already exists, the copy operation merges the
    contents, replacing existing files (controlled by the 'exist_ok' parameter).

    Parameters
    ----------
    src_folder : str
        The path to the source directory.
    dst_folder : str
        The path to the destination directory.
    exist_ok : bool, optional
        If True (default), it allows the copy operation even if dst_folder 
        already exists, merging the contents. If False and dst_folder exists, 
        a FileExistsError is raised.

    Returns
    -------
    Optional[str]
        The path to the destination directory if successful, otherwise None.

    Raises
    ------
    FileNotFoundError
        If the source directory does not exist.
    Exception
        For other operational errors during the copy process.
    """
    
    # 1. Input Validation
    if not os.path.isdir(src_folder):
        raise FileNotFoundError(f"Source directory not found: {src_folder}")

    try:
        # 2. Perform the recursive copy operation
        # shutil.copytree is the standard way to copy directories in Python
        shutil.copytree(
            src=src_folder,
            dst=dst_folder,
            dirs_exist_ok=exist_ok  # Controls behavior if dst_folder exists
        )
        
        # 3. Success message and return value
        print(f"Directory successfully copied from '{src_folder}' to '{dst_folder}'.")
        return dst_folder

    except FileExistsError as e:
        # This error is usually caught only if exist_ok=False and dst_folder exists
        print(f"Error: Destination directory '{dst_folder}' already exists and exist_ok is False.")
        raise e
        
    except Exception as e:
        # Catch any other potential errors (e.g., permission denied, disk full)
        print(f"An error occurred during the copy operation: {e}")
        return None