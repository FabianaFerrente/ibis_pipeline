import numpy as np

def lpff_pol(arr):
    """
    Misura la posizione di una linea spettrale con accuratezza subpixel
    utilizzando il metodo della fase di Fourier.

    Parametri:
    arr : array 1D
        Contiene il profilo della linea spettrale.

    Ritorna:
    pos : float
        Posizione della linea misurata dal bordo sinistro dell'array.
    """
    length = len(arr)
    mid = length / 2.0
    tpi = 2.0 * np.pi  # 2 * pi
    dp = 360.0 / length  # Gradi per pixel

    # Trasformata di Fourier
    fl1 = np.fft.fft(arr)

    # Calcolo della fase e posizione
    lp = -np.arctan2(fl1[1].imag, fl1[1].real) / tpi * 360.0
    pos = lp / dp + mid

    return pos
