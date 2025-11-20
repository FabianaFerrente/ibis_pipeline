
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
#               AVERAGE DARK BB WAVELENGTH
#------------------------------------------
def avg_dark_bb_wavelength(
    directory: str,
    wavelength,
    method: str = 'mad',
    outlier_sigma: float = 3.0
):
    search = '*BB_Dark*.fits'
    
    all_files=ut.find_files_in_directory(directory, search)
    #print(all_files)
    # 2) Seleziona i file per la λ richiesta
    files = ut.get_files_for_wavelength(all_files, wavelength)

    if not files:
        raise ValueError(f"No dark  BB files found for wavelength {wavelength} in {directory}")

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
    
    # 1. Seleziona tutti i frame (dal cubo 'images') che hanno passato l'outlier rejection ('mask')
    good_data = images[mask]
    
    # 2. Calcola la media lungo l'asse dei frame (asse 0) per ottenere un'immagine 2D finale
    dark_mean = good_data.mean(axis=0)

    # Nota: Rimuovi la sezione che ricalcola frame_means, mean_val, std_val e good_frames
    
    return dark_mean, valid_idx, files, dlevel, drms # Modificato per restituire dark_mean 2D



#------------------------------------------
#               DARK BB FITS
#------------------------------------------

def  dark_bb_fits(input_directory: str, output_directory: str, wavelength: int, method, outlier, plot_and_save=True):
    dark_avg, valid_idx, files, dlevel, drms = avg_dark_bb_wavelength(input_directory, wavelength, method=method, outlier_sigma=outlier)
    
    n_frames= len(valid_idx)
    # Plot statistiche concatenate
    if plot_and_save:
        plot_stats(
            {"mean_level": dlevel, "rms": drms},
            wavelength=wavelength,
            calib = 'BB_Dark',
            show=False,
            save_dir = os.path.join(output_directory, 'plot')
        )
    # Salva FITS con dark medio
    output_name = f"{wavelength}_BB_DarkAvg_{n_frames}frames_{outlier:.1f}sigma_new.fits"
    print(output_name)

    output_path = os.path.join(output_directory, output_name)
    # CORREZIONE: Aggiungi una dimensione all'immagine 2D per farla diventare un cubo (1, Ny, Nx)
    dark_avg_3d = np.expand_dims(dark_avg, axis=0) # Forma: (1, Ny, Nx)
    all_files = ut.find_files_in_directory(input_directory, '*BB_Dark*.fits')
    input_fits_files = ut.get_files_for_wavelength(all_files, wavelength)

    fits_io.write_data_cube_as_separate_hdus(input_fits_files, output_path, dark_avg_3d, pipeline_step_info={})
    print(f" Dark medio salvato: {output_path}")
    

#------------------------------------------
#               AVG FLAT BB WAVELENGTH
#------------------------------------------
def avg_flat_bb_wavelength(
    directory: str,
    wavelength,
    dark_image: np.ndarray,
    method: str = 'mad',
    outlier_sigma: float = 0.9 # Il codice IDL usava 0.9 come soglia MINIMA
) -> tuple:
    """
    Calcola l'immagine Flat Field media per una specifica lunghezza d'onda,
    eseguendo la sottrazione del Dark e il rigetto degli outlier.
    
    Parameters
    ----------
    files : list of str
        Percorsi dei file FITS Flat Field per la lunghezza d'onda selezionata.
    dark_image : numpy.ndarray
        L'immagine Dark Field media 2D da sottrarre.
    method : str
        Metodo per l'outlier rejection ('mad', 'std').
    outlier_sigma : float
        Il valore soglia per l'outlier rejection (corrisponde a 0.9 nel codice IDL
        se la deviazione è interpretata come scarto dalla mediana/media).
    
    Returns
    -------
    tuple
        (flat_avg_2d, valid_idx, flat_levels, flat_rms, flat_times)
    """
    
    search = '*BB_FF*.fits'
    
    all_files=ut.find_files_in_directory(directory, search)
    #print(all_files)
    # 2) Seleziona i file per la λ richiesta
    files = ut.get_files_for_wavelength(all_files, wavelength)
    if not files:
        raise ValueError("Nessun file flat field fornito.")
    
    # 3) Estrai statistiche e stack di immagini concatenato
    stats = extract_frames(files)

    flevel_raw = np.array(stats["mean_level"])  # Livello medio di tutti i frame
    images_raw = np.array(stats["images"])      # Cubo 3D di tutti i frame
    
    ntotal = len(flevel_raw)
    
    # --- RIGETTO DEGLI OUTLIER ---
    
    # L'IDL usava: tmp = flevel / median(flevel); MASK = TMP GE 0.9
    
    # 1. Normalizzazione (equivalente a tmp = flevel / median(flevel))
    flevel_median = np.median(flevel_raw)
    if flevel_median == 0:
        raise ValueError("Mediana dei livelli flat è zero. Impossibile normalizzare.")
        
    flevel_norm = flevel_raw / flevel_median
    # 2. Mascheramento (equivalente a MASK = TMP GE 0.9)
    # Rigetta i frame il cui livello medio è inferiore alla soglia
    # NOTA: Usiamo 'mask' per coerenza con la nomenclatura pipeline
    mask = st.reject_outliers(flevel_norm, method=method, threshold=outlier_sigma)

    valid_idx   = np.where(mask)[0].tolist()
    invalid_idx = np.where(~mask)[0].tolist()

    print(f"λ={wavelength} Å: {len(valid_idx)} frame validi, {len(invalid_idx)} scartati")
    if invalid_idx:
        print(f"   Frame scartati (indici globali): {invalid_idx}")

    
    # Alternativa robusta (usando la funzione di rigetto)
    # Se vuoi usare la funzione 'reject_outliers' per un rigetto più standard
    # come faceva la dark pipeline, dovrai adattare la logica qui.
    # Per replicare la logica IDL (TMP GE 0.9), usiamo il confronto diretto.
    ntot = len(valid_idx)
    
    print('------------------------------------------------------')
    print(f'Total number of flat images used: {ntot} (scartati: {ntotal - ntot})')
    print('------------------------------------------------------')

    if ntot == 0:
        raise RuntimeError("Nessun frame flat field valido dopo il rigetto.")

    # --- CALCOLO DELLA MEDIA ---

    # 1. Sottrai il dark da TUTTI i frame validi
    np.expand_dims(dark_image, axis=0) 
    # Seleziona i frame validi
    valid_images = images_raw[mask]
    
    # dark_subtracted_images = valid_images - dark_image 
    # La sottrazione è automatica se le dimensioni (Ny, Nx) coincidono

    # 2. Accumula la somma dei frame validi e calibrati
    # In Python, usiamo la media diretta anziché la somma e divisione finale
    
    # dark_subtracted_images = valid_images - dark_image
    # flat_avg = dark_subtracted_images.mean(axis=0)
    
    # Versione efficiente con un solo passaggio:
    flat_avg = np.mean(valid_images - dark_image, axis=0)
    
    # --- Estrazione dei Dati di Output Filtrati (Optional, per coerenza) ---
    
    flat_levels = flevel_raw[mask]
    flat_rms    = np.array(stats['rms'])[mask]
    flat_times  = np.array(stats['time'])[mask]
    
    # Nota: Il codice IDL divideva flat /= FLOAT(ntotal), ma ntotal è il numero totale di frame.
    # La divisione corretta per la media è per ntot (numero di frame USATI).
    # L'uso di np.mean(..., axis=0) lo gestisce automaticamente.
    
    print('----------------------------')
    print('statistics mean flat :      ')
    print('----------------------------')
    print(f"Mean (output): {np.mean(flat_avg):.4f}, Std: {np.std(flat_avg):.4f}")

    return flat_avg, valid_idx, flat_levels, flat_rms, flat_times 


#------------------------------------------
#               FLAT BB FITS
#------------------------------------------
def flat_bb_fits(
    input_directory: str, 
    output_directory: str, 
    wavelength: int, 
    method: str, 
    outlier: float, 
    dark_image: np.ndarray, 
    plot_and_save: bool = True
):
    """
    Calcola, filtra e salva l'immagine Flat Field media 2D per una specifica lunghezza d'onda 
    (banda larga), sottraendo il Dark Field.
    """
    
    # 1. Calcolo del Flat medio 2D (incl. dark subtraction e outlier rejection)
    # Si noti che la funzione avg_flat_bb_wavelength deve accettare dark_image
    flat_avg, valid_idx, files, flevel, frms = avg_flat_bb_wavelength(
        input_directory, 
        wavelength, 
        method=method, 
        outlier_sigma=outlier,
        dark_image=dark_image 
    )
    
    n_frames = len(valid_idx)
    
    # 2. Plot statistiche concatenate
    if plot_and_save:
        # Nota: La funzione plot_dark_statistics dovrebbe essere rinominata 
        # in plot_flat_statistics per coerenza se usata per flevel/frms
        plot_stats( 
            {"mean_level": flevel, "rms": frms},
            wavelength=wavelength,
            calib= 'BB_Flat',
            show=False,
            save_dir=os.path.join(output_directory, 'plot')
        )
    
    # 3. Preparazione per il salvataggio FITS
    # Aggiornamento nome file
    output_name = f"{wavelength}_BB_FlatAvg_{n_frames}frames_{outlier:.1f}sigma_new.fits"
    print(output_name)

    output_path = os.path.join(output_directory, output_name)
    
    # 4. Ottenimento dei percorsi dei file Flat originali (per l'header template)
    # Aggiornamento pattern di ricerca
    all_files = ut.find_files_in_directory(input_directory, '*BB_FF*.fits')
    input_fits_files = ut.get_files_for_wavelength(all_files, wavelength)

    # 5. Scrittura del FITS (Convertiamo il 2D in un cubo 3D con un solo frame)
    flat_avg_3d = np.expand_dims(flat_avg, axis=0) # Forma: (1, Ny, Nx)

    fits_io.write_data_cube_as_separate_hdus(
        input_fits_files, 
        output_path, 
        flat_avg_3d, 
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