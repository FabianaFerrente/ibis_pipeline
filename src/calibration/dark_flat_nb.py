
import os
import glob
import numpy as np
from astropy.io import fits
from astropy.io.fits import ImageHDU # Per tipizzazione e verifica
import matplotlib.pyplot as plt

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils import utils_io as ut
from utils import fits_io as fits_io
from utils import time_utils as time_utils
from utils import stats as st
from utils.plotting import plot_statistics as plot_stats


from typing import List, Dict, Any

import numpy as np
from astropy.io import fits

#------------------------------------------
#               EXTRACT FRAMES
#------------------------------------------
def extract_frames(files: List[str]) -> Dict[str, Any]:
    """
    Extract statistics and images from a list of dark frame FITS files,
    where each file contains dark frames as separate ImageHDU extensions.

    This function is designed to handle FITS files where individual dark frames are stored
    as **separate ImageHDU extensions** (starting from HDU[1]), rather than as a single 3D data cube.
    It calculates basic statistics (mean signal, relative RMS) for each frame and extracts
    its associated observation time.
    
    Parameters
    ----------
    files : list of str
        List of file paths to FITS files containing dark frames. Each file
        must contain dark frames in subsequent ImageHDU extensions (HDU[1], HPU[2], ...).

    :raises FileNotFoundError: If an input file path does not exist.
    :raises Exception: For general errors encountered during FITS file processing (e.g., corrupted file).
    :returns: A dictionary containing concatenated statistics and data across all input files.
    :rtype: dict
    
    Returns
    -------
    result : dict
        Dictionary with concatenated statistics and data (keys unchanged from original).
    :mean_level: Concatenated mean signal level for every extracted frame.
    :type mean_level: list[float]
    :rms: Concatenated relative Root Mean Square (standard deviation / mean), or :any:`numpy.nan` if the mean is zero.
    :type rms: list[float]
    :time: Observation times in seconds (relative to midnight), one entry per frame. Time is extracted from the 
           :literal:`DATE-BEG` keyword in the Primary Header (HDU[0]) of the corresponding FITS file.
    :type time: list[float]
    :images: A list of all 2D dark frames (:class:`numpy.ndarray`), concatenated in sequential order.
    :type images: list[:class:`numpy.ndarray`]
    :n_files: The total number of input FITS files processed.
    :type n_files: int
    :frames_per_file: The number of frames successfully extracted from each input FITS file.
    :type frames_per_file: list[int]
    """
    mean_levels: List[float] = []
    rms_values: List[float] = []
    times: List[float] = []
    images: List[np.ndarray] = []
    frames_per_file: List[int] = []

    for fname in files:
        seconds = 0.0 # Valore di default in caso di fallimento
        current_file_frames = 0
        
        try:
            with fits.open(fname) as hdul:
                
                # --- 1. Estrazione dell'ora dal Primary Header (HDU[0]) ---
                primary_header = hdul[0].header
                date_obs = primary_header.get("DATE-BEG", None)
                
                if date_obs:
                    try:
                        seconds = time_utils.time_to_seconds(date_obs)
                    except ValueError:
                        # Se time_to_seconds solleva un ValueError, si mantiene seconds = 0.0
                        pass 
                
                # --- 2. Iterazione sulle ImageHDU (Inizio da HDU[1]) ---
                for hdu in hdul[1:]:
                    # Verifica se l'HDU è un'immagine 2D e contiene dati
                    if isinstance(hdu, ImageHDU) and hdu.data is not None and hdu.data.ndim == 2:
                        
                        frame = hdu.data.astype(np.float32)
                        
                        # Calcolo Statistiche
                        mean_val = np.mean(frame)
                        mean_levels.append(mean_val)
                        
                        # RMS relativo (std / mean)
                        rms_values.append(np.std(frame) / mean_val if mean_val != 0 else np.nan)
                        
                        # L'ora è la stessa (seconds) per tutti i frame estratti da questo file
                        times.append(seconds)
                        
                        images.append(frame)
                        current_file_frames += 1

        except FileNotFoundError:
            print(f"ATTENZIONE: File non trovato e saltato: {fname}")
            continue
        except Exception as e:
            print(f"ERRORE: Impossibile processare il file {fname}. Errore: {e}")
            continue

        # --- 3. Salvataggio del conteggio ---
        frames_per_file.append(current_file_frames)

    return {
        "mean_level": mean_levels,
        "rms": rms_values,
        "time": times,
        "images": images,
        "n_files": len(files),
        "frames_per_file": frames_per_file,
    }
    
#------------------------------------------
#               AVERAGE DARK NB WAVELENGTH
#------------------------------------------
def avg_dark_nb_wavelength(
    directory: str,
    wavelength,
    method: str = 'mad',
    outlier_sigma: float = 3.0
):
    """
    Compute the averaged dark cube for a single wavelength in a directory,
    performing outlier rejection on a frame-by-frame basis.

    For the requested wavelength, the function:

    1. Gathers input FITS dark files at that λ.
    2. Extracts statistics with :func:`extract_frames`.
    3. Applies outlier rejection on the mean signal level of each frame.
    4. Builds an averaged dark cube preserving the frame index structure
       (i.e. frame *i* corresponds to the average across all valid files).
    5. Returns the averaged cube, indices of valid frames (global, concatenated),
       and the list of input FITS files used.

    Parameters
    ----------
    directory : str
        Path to the directory containing dark FITS files.
    wavelength : int or float or str
        Wavelength (Å) to process.
    method : str, optional
        Method for outlier rejection (e.g., ``'std'`` or ``'mad'``).
        Passed to :func:`reject_outliers`. Default is ``'mad'``.
    outlier_sigma : float, optional
        Sigma/threshold parameter for outlier rejection on mean levels.
        Default is 3.0.

    Returns
    -------
    dark_avg : numpy.ndarray
        Averaged dark cube with shape ``(n_frames, ny, nx)``,
        where *n_frames* is the number of frames per input file.
        Frames with no valid data are filled with NaN.
    valid_idx : list of int
        Global indices (sull'array concatenato) dei frame che hanno passato
        l'outlier rejection.
    files : list of str
        List of input FITS file paths for the requested wavelength.

    Notes
    -----
    - Assumes all input files for the given wavelength contain the same
      number of frames.
    - Frame *i* del cubo medio è la media dei frame *i* di tutti i file
      che non sono stati esclusi come outlier (sulla metrica del mean level).
    - Gli indici in ``valid_idx`` e i log stampati sono riferiti alla
      concatenazione dei frame: [file0_f0, file0_f1, ..., file1_f0, file1_f1, ...].

    Examples
    --------
    >>> dark_avg, valid_idx, files = avg_dark_nb_all(
    ...     "./data/darks/", wavelength=6173, method="mad", outlier_sigma=3.0
    ... )
    λ=6173 Å: 45 frame validi, 5 scartati
    """
    # 1)
    
    search = '*NB_Dark*.fits'
    
    all_files=ut.find_files_in_directory(directory, search)
    #print(all_files)
    # 2) Seleziona i file per la λ richiesta
    files = ut.get_files_for_wavelength(all_files, wavelength)

    if not files:
        raise ValueError(f"No dark  NB files found for wavelength {wavelength} in {directory}")

    # 3) Estrai statistiche e stack di immagini concatenato
    stats = extract_frames(files)

    dlevel = np.array(stats["mean_level"])   # (N_tot_frames,)
    drms   = np.array(stats["rms"])        
    images = np.array(stats["images"])       # (N_tot_frames, Y, X)
    n_files = stats["n_files"]
    frames_per_file = stats["frames_per_file"]

    # 4) Outlier rejection sui mean level concatenati
    mask = st.reject_outliers(dlevel, method=method, threshold=outlier_sigma)

    valid_idx   = np.where(mask)[0].tolist()
    invalid_idx = np.where(~mask)[0].tolist()

    print(f"λ={wavelength} Å: {len(valid_idx)} frame validi, {len(invalid_idx)} scartati")
    if invalid_idx:
        print(f"   Frame scartati (indici globali): {invalid_idx}")

    # 5) Se nessun frame valido, esci
    if len(valid_idx) == 0:
        raise RuntimeError(f"Nessun frame valido dopo outlier rejection per λ={wavelength} Å")

    # 6) Ricostruisci il cubo medio mantenendo la struttura frame-per-frame
    #    (si assume numero di frame per file costante)
    if len(set(frames_per_file)) != 1:
        raise ValueError(
            f"I file a λ={wavelength} hanno numero di frame non uniforme: {frames_per_file}"
        )
    n_frames = frames_per_file[0]
    ny, nx = images.shape[1:] # solo le dimensioni spaziali (frame concatenati)
    dark_avg = np.zeros((n_frames, ny, nx), dtype=np.float32) # cube 3D(n_frames, ny, nx)

    # media per frame index i (prendendo i frame i-esimi di ciascun file validi)
    for i in range(n_frames):
        # indici globali dei frame i-esimi: i, i+n_frames, i+2*n_frames, ...
        idx = np.arange(i, len(dlevel), n_frames)
        idx_validi_i = [j for j in idx if mask[j]]
        if idx_validi_i:
            dark_avg[i] = np.mean(images[idx_validi_i], axis=0)
        else:
            dark_avg[i] = np.nan

    return dark_avg, valid_idx, files, dlevel, drms

#------------------------------------------
#               DARK NB FITS
#------------------------------------------
def  dark_nb_fits(input_directory: str, output_directory: str, wavelength: int, method, outlier, plot_and_save=True):
    """
    Compute a mean dark frame from multiple dark calibration FITS files,
    removing outlier frames that deviate significantly from the mean.

    Parameters
    ----------
    input_directory : str
        Path to the directory containing dark FITS files.
    output_directory : str
        Path to the directory where output FITS files will be saved.
    wavelength : int
        Wavelength (Å) of the dark FITS files to process.
    method : str
        Method for outlier rejection (e.g., ``'std'`` or ``'mad'``).
    outlier_sigma : float
        Sigma/threshold parameter for outlier rejection on mean levels.
    plot_and_save : bool, optional
        If True, plot statistics of the concatenated frames and save the plots.
        Default is True.

    Returns
    -------
    None
    """
    dark_avg, valid_idx, files, dlevel, drms = avg_dark_nb_wavelength(input_directory, wavelength, method=method, outlier_sigma=outlier)
    n_frames= len(valid_idx)
    # Plot statistiche concatenate
    if plot_and_save:
        plot_stats(
            {"mean_level": dlevel, "rms": drms},
            wavelength=wavelength,
            calib = 'NB_Dark',
            show=False,
            save_dir = os.path.join(output_directory, 'plot')
        )
    # Salva FITS con dark medio
    output_name = f"{wavelength}_NB_DarkAvg_{n_frames}frames_{outlier:.1f}sigma_new.fits"
    print(output_name)

    output_path = os.path.join(output_directory, output_name)
    
    all_files = ut.find_files_in_directory(input_directory, '*NB_Dark*.fits')
    input_fits_files = ut.get_files_for_wavelength(all_files, wavelength)

    fits_io.write_data_cube_as_separate_hdus(input_fits_files, output_path, dark_avg, pipeline_step_info={})
    print(f" Dark medio salvato: {output_path}")




import numpy as np
from typing import List, Dict, Any, Tuple
# Assumiamo che st, ut, extract_flat_frames siano importati

def avg_flat_nb_wavelength(
    directory: str,
    wavelength: int,
    dark_image: np.ndarray, # L'immagine Dark 2D da sottrarre
    method: str = 'mad',
    outlier_sigma: float = 3.0
) -> Tuple[np.ndarray, List[int], List[str], np.ndarray, np.ndarray]:
    
    search = '*NB_Flat*.fits'
    
    # 1. Trova e filtra i file per la lambda richiesta
    all_files = ut.find_files_in_directory(directory, search)
    files = ut.get_files_for_wavelength(all_files, wavelength)

    if not files:
        raise ValueError(f"No flat NB files found for wavelength {wavelength} in {directory}")

    # 2. Estrai statistiche e stack di immagini concatenato
    # (Usiamo una funzione di estrazione dati che restituisce tutti i frame concatenati)
    stats = extract_frames(files)

    flevel = np.array(stats["mean_level"])   # Livelli medi concatenati (N_tot_frames,)
    frms   = np.array(stats["rms"])        
    images = np.array(stats["images"])       # Cubo concatenato (N_tot_frames, Y, X)
    frames_per_file = stats["frames_per_file"]
    n_files = stats["n_files"]
    # 3. Rigetto degli Outlier sui mean level concatenati
    # (Questo è il rigetto globale, ma verrà applicato localmente nella media)
    
    # NOTA: Per replicare la logica IDL (fflevel / median(fflevel) >= 0.9), 
    # potresti dover adattare o sostituire st.reject_outliers qui, ma manteniamo
    # il rigetto basato su MAD/STD per robustezza se outlier_sigma è diverso da 0.9.
    mask = st.reject_outliers(flevel, method=method, threshold=outlier_sigma)

    valid_idx   = np.where(mask)[0].tolist()
    invalid_idx = np.where(~mask)[0].tolist()

    print(f"λ={wavelength} Å: {len(valid_idx)} frame validi, {len(invalid_idx)} scartati")
    if invalid_idx:
        print(f"   Frame scartati (indici globali): {invalid_idx}")

    if len(valid_idx) == 0:
        raise RuntimeError(f"Nessun frame valido dopo outlier rejection per λ={wavelength} Å")

    # 4. Ricostruisci il Cubo Flat Medio 3D (Frame-per-Frame)
    
    if len(set(frames_per_file)) != 1:
        raise ValueError(
            f"I file a λ={wavelength} hanno numero di frame non uniforme: {frames_per_file}"
        )
        
    # n_frames è il numero di canali/scans per file (es. 20 canali spettrali)
    n_frames = frames_per_file[0] 
    ny, nx = images.shape[1:] 
    
    # Cubo 3D di output: (N_channels, Ny, Nx)
    flat_avg_cube = np.zeros((n_frames, ny, nx), dtype=np.float32)

    # Assicurati che dark_image sia un cubo 3D
    if dark_image.ndim != 3 or dark_image.shape[0] != n_frames:
        raise ValueError(
            f"Dark image deve essere un cubo 3D con {n_frames} canali, ma ha forma {dark_image.shape}"
        )

    # Media per frame index i (i-esima posizione/canale)
    for i in range(n_frames):
        # 1. Seleziona il frame Dark 2D corrispondente al canale i
        dark_frame_i = dark_image[i, :, :] 
        
        # Indici globali che puntano alla posizione i
        idx = np.arange(i, len(flevel), n_frames)
        
        # Filtra solo gli indici che hanno superato l'outlier rejection
        idx_validi_i = [j for j in idx if mask[j]]
        
        if idx_validi_i:
            # 2. Seleziona i frame validi per la posizione 'i' (cubo 3D: N_validi x Ny x Nx)
            frames_to_average = images[idx_validi_i] 
            
            # 3. Sottrai il Dark_frame_i (2D) a tutti i frame validi (broadcasting)
            calibrated_frames = frames_to_average - dark_frame_i 
            
            # 4. Calcola la media lungo l'asse dei frame
            flat_avg_cube[i] = np.mean(calibrated_frames, axis=0)
        else:
            flat_avg_cube[i] = np.nan
            
    return flat_avg_cube, valid_idx, files, flevel, frms



import os
# ... (altre importazioni)

def flat_nb_fits(
    input_directory: str, 
    output_directory: str, 
    wavelength: int, 
    method: str, 
    outlier: float, 
    dark_image: np.ndarray, # Immagine Dark 2D
    plot_and_save: bool = True
):
    
    # Calcola il Flat Medio 3D (Cubo)
    flat_avg_cube, valid_idx, files, flevel, frms = avg_flat_nb_wavelength(
        input_directory, 
        wavelength, 
        dark_image, 
        method=method, 
        outlier_sigma=outlier
    )
    
    n_frames = len(valid_idx)
    
    # Plot statistiche (flevel e frms sono le statistiche sui frame individuali validi)
    if plot_and_save:
        plot_stats( 
            {"mean_level": flevel, "rms": frms},
            wavelength=wavelength,
            calib = 'NB_Flat',
            show=False,
            save_dir=os.path.join(output_directory, 'plot')
        )
    
    # Salvataggio FITS (flat_avg_cube è già 3D, non serve expand_dims)
    output_name = f"{wavelength}_NB_FlatAvg_{n_frames}frames_{outlier:.1f}sigma_new.fits"
    print(output_name)

    output_path = os.path.join(output_directory, output_name)
    
    all_files = ut.find_files_in_directory(input_directory, '*NB_FF*.fits')
    input_fits_files = ut.get_files_for_wavelength(all_files, wavelength)


    # Scrive il cubo 3D (N_channels, Ny, Nx) nel FITS multi-HDU
    fits_io.write_data_cube_as_separate_hdus(
        input_fits_files, 
        output_path, 
        flat_avg_cube, # E' già 3D
        pipeline_step_info={}
    )
    print(f" Flat medio salvato: {output_path}")    

#------------------------------------------
#               MAIN
#------------------------------------------
def main():
    # Example usage
    dir_dark = "/Users/giovanna/IBIS/ibis_pipeline_python/ibis_pipeline_definitive/data/input/MEF"  # Replace with your data directory
    directory = "/Users/giovanna/IBIS/ibis_pipeline_python/ibis_pipeline_definitive/data/input/MEF"  # Replace with your data directory
    output_dir = "/Users/giovanna/IBIS/ibis_pipeline_python/ibis_pipeline_definitive/data/output/"  # Replace with your desired output directory
    dark_avg, valid_idx, files, dlevel, drms = avg_dark_bb_wavelength(dir_dark, wavelength=6173, method='mad', outlier_sigma=3.0)
    

    #flat_bb_fits(directory, output_dir, wavelength=6173, method='mad', outlier=3.0, dark_image = dark_avg, plot_and_save=True)
    fits_io.inspect_fits(os.path.join(output_dir, '6173_BB_FlatAvg_240frames_3.0sigma_new.fits'))
    exit()
    dark_avg, valid_idx, files, dlevel, drms = avg_dark_nb_all_wavelength(directory, wavelength=6173, method='mad', outlier_sigma=3.0)
    print(dark_avg.shape)
    exit()
    
    darkNB_files = sorted(glob.glob(os.path.join(directory, '*NB_Dark*.fits')))


    for input_fits in darkNB_files:
        output_fits = input_fits.replace('.fits', '_MEF.fits')
        print(output_fits)
        fits_io.convert_fits_to_mef(input_fits, output_fits)
        
if __name__ == "__main__":
    main()