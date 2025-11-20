import numpy as np
from astropy.io import fits

def inspect_fits(filename: str):
    """
    Ispeziona un file FITS e stampa:
    - Informazioni sugli HDU
    - Header del Primary HDU
    - Header degli Image HDU
    """
    with fits.open(filename) as hdul:
        print(f"\nFile: {filename}")
        print("Struttura del FITS (HDU list):")
        hdul.info()

        print("\n=== HEADER PRIMARIO ===")
        print(repr(hdul[0].header))  # header completo del Primary HDU

        # Itera sugli altri HDU
        for i, hdu in enumerate(hdul[1:], start=1):
            if isinstance(hdu, fits.ImageHDU):
                print(f"\n=== HEADER IMAGE HDU {i} ===")
                print(repr(hdu.header))  # header dell'immagine
            elif isinstance(hdu, fits.BinTableHDU):
                print(f"\n=== HEADER BINTABLE HDU {i} ===")
                print(repr(hdu.header))
            elif isinstance(hdu, fits.TableHDU):
                print(f"\n=== HEADER TABLE HDU {i} ===")
                print(repr(hdu.header))

def convert_fits_to_mef(input_fits, output_fits):
    # === Leggi il file FITS 3D ===
    with fits.open(input_fits) as hdul:
        data_cube = hdul[0].data       # (ny, nx, nframes)
        primary_hdr = hdul[0].header

    nframes, nx, ny  = data_cube.shape
    print(f"Cube shape: {data_cube.shape}")

    # === Crea PrimaryHDU (solo header) ===
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header = primary_hdr
    primary_hdu.header["EXTNAME"] = "DARK"
    primary_hdu.header["NFRAMES"] = nframes

    # === Crea le ImageHDU per ogni frame ===
    image_hdus = []
    for i in range(nframes):
        frame_data = data_cube[i, :, :]
        hdu = fits.ImageHDU(frame_data.astype(np.int32), name=f"Image{i}")
        hdu.header["FRAMEID"] = i
        hdu.header["EXTNAME"] = f"Image{i}"
        image_hdus.append(hdu)

    # === Costruisci l'HDUList finale ===
    hdul_new = fits.HDUList([primary_hdu] + image_hdus)

    # === Salva il nuovo FITS ===
    hdul_new.writeto(output_fits, overwrite=True)
    print(f" Saved MEF FITS: {output_fits}")
    
                   
def extract_all_headers_with_comments(fits_file):
    """
    Extract the Primary HDU header and all subsequent Image HDU headers from a FITS file,
    returning them as a dictionary (primary) and a list of dictionaries (images),
    where each keyword maps to (value, comment).

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.

    Returns
    -------
    primary_dict : dict
        Dictionary {keyword: (value, comment)} from the Primary HDU header.
    image_dicts : list of dict
        List of dictionaries [{keyword: (value, comment)}, ...], one for each Image HDU.
    """
    with fits.open(fits_file) as hdul:
        # --- Primary header
        primary_hdr = hdul[0].header
        primary_dict = {
            card.keyword: (card.value, card.comment)
            for card in primary_hdr.cards
        }

        # --- List of headers for all subsequent HDUs
        image_dicts = []
        for idx, hdu in enumerate(hdul[1:], start=1):  # skip primary
            hdr_dict = {
                card.keyword: (card.value, card.comment)
                for card in hdu.header.cards
            }
            image_dicts.append(hdr_dict)

    return primary_dict, image_dicts

               
def extract_headers_with_comments(fits_file, image_ext=1):
    """
    Extract Primary and Image HDU headers from a FITS file as dictionaries,
    including value and comment for each keyword.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.
    image_ext : int, optional
        Extension index of the image HDU (default=1).

    Returns
    -------
    primary_dict : dict
        Dictionary {keyword: (value, comment)} from the Primary HDU header.
    image_dict : dict
        Dictionary {keyword: (value, comment)} from the selected Image HDU header.
    """
    with fits.open(fits_file) as hdul:
        primary_hdr = hdul[0].header
        primary_dict = {card.keyword: (card.value, card.comment) for card in primary_hdr.cards}

        if image_ext < len(hdul):
            image_hdr = hdul[image_ext].header
            image_dict = {card.keyword: (card.value, card.comment) for card in image_hdr.cards}
        else:
            image_dict = {}  # se non c'è l'estensione, restituisce dict vuoto

    return primary_dict, image_dict



def dict_to_header(hdict):
    """
    Convert a dictionary {keyword: (value, comment)} into a FITS Header.

    Parameters
    ----------
    hdict : dict
        Dictionary where keys are FITS keywords and values are (value, comment).

    Returns
    -------
    fits.Header
        FITS Header object.
    """
    hdr = fits.Header()
    for key, (val, com) in hdict.items():
        hdr[key] = (val, com)
    return hdr

def create_fits_from_dicts(output_file, primary_dict, images, image_dicts):
    """
    Create a FITS file from header dictionaries and image data.

    Parameters
    ----------
    output_file : str
        Path where the FITS file will be written.
    primary_dict : dict
        Dictionary {keyword: (value, comment)} for the Primary Header.
    images : list of numpy.ndarray
        List of image arrays for each ImageHDU.
    image_dicts : list of dict
        List of dictionaries {keyword: (value, comment)} for each ImageHDU.

    Returns
    -------
    None
        The FITS file is written to disk.
    """
    # Primary HDU
    primary_hdr = dict_to_header(primary_dict)
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)

    # Image HDUs
    image_hdus = []
    for img, hdict in zip(images, image_dicts):
        hdr = dict_to_header(hdict)
        image_hdus.append(fits.ImageHDU(data=img, header=hdr))

    # HDUList
    hdul = fits.HDUList([primary_hdu] + image_hdus)
    hdul.writeto(output_file, overwrite=True)
    print(f"✅ FITS creato: {output_file}")
    
    
import os
import numpy as np
from astropy.io import fits
from typing import List, Dict, Any, Tuple, Optional
   
def _prepare_fits_headers_old(
    template_path: str,
    input_paths: List[str],
    output_path: str,
    pipeline_info: Dict[str, Any],
    remove_keys: Optional[List[str]]
) -> Tuple[fits.Header, Dict[str, Any]]:
    """
    Generates the Primary and Image headers for the output FITS file.
    """
    
    # Eredita gli header dal file template
    primary_dict, image_dict = extract_headers_with_comments(
        template_path,
        image_ext=1
    )

    # 1. Rimuovi le keywords indesiderate (CLEANUP)
    if remove_keys:
        for key in remove_keys:
            primary_dict.pop(key, None)
            image_dict.pop(key, None)

    # 2. Aggiorna l'Header Principale (PRIMARY HDU)
    primary_dict['FILENAME'] = (os.path.basename(output_path), 'Output file name')
    primary_dict['NINPUTS'] = (len(input_paths), "Number of parent FITS files used")
    
    # Aggiunge i nomi dei file input
    for i, fname in enumerate(input_paths, start=1):
        key = f"INPUT_N{i}"
        value = os.path.basename(fname)
        primary_dict[key] = (value, f"Parent FITS file #{i}")

    # Aggiunge le informazioni specifiche della pipeline
    for key, (value, comment) in pipeline_info.items():
        primary_dict[key] = (value, comment)

    # Converti il Primary Dict in un oggetto Header FITS
    primary_hdr = dict_to_header(primary_dict)
    
    # 3. Aggiorna l'Header dell'Immagine (IMAGE HDU) - Aggiungi info generiche
    image_dict['NINPUTS'] = (len(input_paths), "Number of parent FITS files")

    # Ritorna l'Header FITS finale per il Primary HDU e il Dict template per gli Image HDU
    return primary_hdr, image_dict



def _prepare_fits_headers(
    template_path: str,
    input_paths: List[str],
    output_path: str,
    pipeline_info: Dict[str, Any],
    remove_keys: Optional[List[str]] = None):
    """
    Generates the Primary and Image headers for the output FITS file,
    starting from a template FITS that may contain multiple ImageHDUs.

    Parameters
    ----------
    template_path : str
        Path to the FITS template file.
    input_paths : list of str
        List of FITS input file paths used to build the output.
    output_path : str
        Path of the output FITS file to be generated.
    pipeline_info : dict
        Dictionary {keyword: (value, comment)} with pipeline metadata.
    remove_keys : list of str, optional
        FITS keywords to remove from both primary and image headers.

    Returns
    -------
    primary_hdr : fits.Header
        Final Primary header object ready to be written.
    image_dicts : list of dict
        List of updated dictionaries for each Image HDU header.
    """
    # === 1. Estrai gli header dal template ===
    primary_dict, image_dicts = extract_all_headers_with_comments(template_path)
    # Prende il primo Image Header come template di base (se la lista non è vuota)
    base_image_dict = image_dicts[0] if image_dicts else {}
    
    # === 2. Rimuovi keywords indesiderate (pulizia) ===
    if remove_keys:
        for key in remove_keys:
            primary_dict.pop(key, None)
            
            for hdr in image_dicts:
                hdr.pop(key, None)

    # === 3. Aggiorna l’header principale (PRIMARY HDU) ===
    primary_dict['FILENAME'] = (os.path.basename(output_path), 'Output FITS file name')
    primary_dict['NINPUTS'] = (len(input_paths), "Number of input FITS files used")

    # Aggiunge i nomi dei file input
    for i, fname in enumerate(input_paths, start=1):
        key = f"INPUT_N{i}"
        value = os.path.basename(fname)
        primary_dict[key] = (value, f"Parent FITS file #{i}")

    # Aggiunge le informazioni specifiche della pipeline
    for key, (value, comment) in pipeline_info.items():
        primary_dict[key] = (value, comment)

    # === 4. Converte il Primary Dict in Header FITS ===
    primary_hdr = dict_to_header(primary_dict)

    # === 5. Aggiorna gli header delle immagini ===
    updated_image_dicts = []
    for i, hdr in enumerate(image_dicts):
        hdr['NINPUTS'] = (len(input_paths), "Number of parent FITS files")
        hdr['PARENT'] = (os.path.basename(template_path), "Template source FITS file")
        hdr['EXTNUM'] = (i + 1, "Extension index")
        updated_image_dicts.append(hdr)

    # === 6. Ritorna il Primary header e la lista aggiornata di header per le immagini ===
    return primary_hdr, updated_image_dicts



    
#--------------------------------------------------
import os
import numpy as np
from astropy.io import fits
from typing import List, Dict, Any, Tuple, Optional



def write_data_cube_as_separate_hdus(
    input_paths: List[str],
    output_path: str,
    data_cube: np.ndarray,
    pipeline_step_info: Dict[str, str],
    remove_keys: Optional[List[str]] = None
) -> None:
    """
    Writes a 3D data cube (n_frames, ny, nx) into a FITS file where each 2D
    frame is stored as a separate ImageHDU, inheriting headers from input files.

    Parameters
    ----------
    input_paths : list of str
        Full paths of the input FITS files used to generate the data_cube.
        The first file is used as the header template.
    output_path : str
        Path where the final FITS file will be saved.
    data_cube : numpy.ndarray
        The data cube to write, shape (n_frames, ny, nx).
    pipeline_step_info : dict
        Dictionary containing pipeline-specific metadata (e.g., version, step name,
        number of frames used, etc.) to be added to the Primary Header.
        Example: {'PIPE_VER': '1.0', 'PRSTEP1': 'AVG-OT', 'NOFUSED': 10}.
    remove_keys : list of str, optional
        Keywords to remove from the inherited headers (both Primary and Image).

    Raises
    ------
    ValueError
        If input_paths is empty or data_cube is not 3D.
    """
    if not input_paths:
        raise ValueError("Input file paths list cannot be empty.")
    if data_cube.ndim != 3:
        raise ValueError(f"Data cube must be 3D (n_frames, ny, nx), got {data_cube.ndim}D.")

    # 1. Prepare Headers 
    primary_hdr, image_hdrs = _prepare_fits_headers(
        template_path=input_paths[0],
        input_paths=input_paths,
        output_path=output_path,
        pipeline_info=pipeline_step_info,
        remove_keys=remove_keys
    )

    # 2. Primary HDU
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)

    # 3. Build Image HDUs for each frame
    n_frames = data_cube.shape[0]
    image_hdus = []
    for i in range(n_frames):
        frame = data_cube[i, :, :]

        # Get the base header for this frame
        frame_hdr = image_hdrs[i].copy()

        # Add frame-specific metadata
        frame_hdr['EXTNAME'] = (f"Image{i:03d}", "Frame index")
        frame_hdr['FRAMIDX'] = (i, "Frame number in sequence")
        frame_hdr['FLUCTUAT'] = (float(np.std(frame)), "Fluctuation of frame data")
        frame_hdr['MEAN_O']   = (float(np.mean(frame)), "Mean of output frame")

        # Aggiorna le dimensioni dell'immagine (NAXIS1/NAXIS2)
        frame_hdr['NAXIS1'] = (frame.shape[1], "Length of axis 1")
        frame_hdr['NAXIS2'] = (frame.shape[0], "Length of axis 2")
        # NOTE: np.float32 is standard for image data in FITS
        # Conversione finale del dict in fits.Header e creazione dell'HDU
        image_hdus.append(fits.ImageHDU(data=frame.astype(np.float32), header=dict_to_header(frame_hdr)))

    # 4. Write Final FITS
    hdul = fits.HDUList([primary_hdu] + image_hdus)
    hdul.writeto(output_path, overwrite=True)

    print(f"FITS file saved with {n_frames} ImageHDU extensions: {output_path}")
    
import glob  
def main():
    dir1 = '/Users/giovanna/IBIS/ibis_pipeline_python/ibis_pipeline_definitive/data/input/flat'
    ff_files = sorted(glob.glob(os.path.join(dir1, '*BB_FF*.fits')))


    for input_fits in ff_files:
        output_fits = input_fits.replace('.fits', '_MEF.fits')
        #print(output_fits)

        convert_fits_to_mef(input_fits, output_fits)
    
        
    
if __name__ == "__main__":
    main()