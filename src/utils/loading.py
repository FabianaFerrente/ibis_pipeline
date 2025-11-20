import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.io import readsav


def load_preprocessed_inputs(flat_file, dark_file, offset_file, prefilter_file):
    """
    Carica dati pre-processati da file FITS:
    - flat, dark: array 3D
    - offset: dizionario con mappe di shift spettrale
    - prefilter: vettore o immagine 1D/3D (correzione spettro)

    Parameters
    ----------
    flat_file : str
        Percorso al FITS contenente il flat calibrato (es. 'output/flat_avg.fits')
    dark_file : str
        Percorso al FITS contenente il dark medio (es. 'output/dark_avg.fits')
    offset_file : str
        Percorso al FITS contenente le mappe di shift spettrale in estensioni (1 per metodo)
    prefilter_file : str
        Percorso al FITS contenente la funzione di trasmissione del prefilter

    Returns
    -------
    flat : ndarray
        Cubo flat (3D)
    dark : ndarray
        Immagine dark media (2D)
    offset : dict
        Dizionario con metodi di shift → mappa 2D (es. {'poly_fit': ..., 'cog': ...})
    prefilter : dict
        dict
    """
    print(f"Loading: flat from {flat_file}")
    with fits.open(flat_file) as hdul:
        flat = hdul[0].data.astype(np.float32)

    print(f"Loading: dark from {dark_file}")
    with fits.open(dark_file) as hdul:
        dark = hdul[0].data.astype(np.float32)

    print(f"Loading: offset maps from {offset_file}")
    #offset = {}
    offset = 1
    """
    with fits.open(offset_file) as hdul:
        for hdu in hdul[1:]:  # skip primary if empty
            key = hdu.name.lower()
            offset[key] = hdu.data.astype(np.float32)
    if not offset:
        raise ValueError(" No offset maps found in extensions!")
    """
    print(f"Loading: prefilter from {prefilter_file}")
    #with fits.open(prefilter_file) as hdul:
        #prefilter = hdul[0].data.astype(np.float32)
    prefilter = readsav(prefilter_file, python_dict=True)

    return flat, dark, offset, prefilter