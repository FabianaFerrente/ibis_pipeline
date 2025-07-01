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
    #lp = -np.arctan2(fl1[1].imag, fl1[1].real) / tpi * 360.0
    
    # Please verify if the line below does the same thing as the line above
    
    lp = -np.arctan2(fl1[1].imag, fl1[1].real, dtype=float) * 180.0 / np.pi
    
    pos = lp / dp + mid

    """
    # Please also verify if this line yields the same results as the code above.
    
    pos = (1.0-np.arctan2(fl1[1].imag, fl1[1].real, dtype=float)/np.pi)*(len(arr)/2.0)
    
    # If yes, I prefer to use this line in the final code, given that this code is called too many times, and using a one-liner is better for performance gains here.
    # I would still keep the rest of the code, comment it out, for explaining purposes
    """

    return pos
