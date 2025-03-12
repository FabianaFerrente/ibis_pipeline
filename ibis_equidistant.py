import numpy as np

def ibis_equidistant(input_str, Nwave, wrange):
    """
    Python equivalent of IDL function `ibis_equidistant`.
    
    Parameters:
    - input_str: Object containing 'grid' (assumed to be an array).
    - wscale: Original wavelength scale.
    - wrange: Optional range [start, end] for wavelength selection.
    
    Returns:
    - wlambda_equi: Equidistant wavelength scale.
    """
    print("Shape of grid:", (input_str.grid).shape)
    print("Contents of grid:", input_str.grid)
    wlambda = np.unique(np.ravel(input_str.grid))
    wlambda.sort()
    print("Shape of wlambda:", wlambda.shape)
    # Correct the double first point issue
    if len(wlambda) < Nwave:
        wlambda1 = np.zeros(Nwave, dtype=wlambda.dtype)
        wlambda1[0] = np.min(wlambda)   # Assicura un valore scalare
        wlambda1[1:Nwave] = wlambda[:Nwave-1]   # Copia gli altri elementi
        wlambda = wlambda1    # Aggiorna wlambda

    # Apply range selection if provided
    if wrange is not None:
        wlambda = wlambda[wrange[0]:wrange[1]]
        
    print("Shape of wlambda:", wlambda.shape)
    print("Contents of wlambda:", wlambda)
    
    if wlambda.size > 0:  # Controlla che wlambda non sia vuoto
        wlambda_f = wlambda[0]  # Prende solo il primo array
        nwlambda = round((np.max(wlambda_f) - np.min(wlambda_f)) / 0.01)
    else:
        nwlambda = 0  # O un altro valore sensato

    wlambda_equi = np.linspace(np.min(wlambda_f), np.max(wlambda_f), nwlambda)

    return wlambda,wlambda_equi
