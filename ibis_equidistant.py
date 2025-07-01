import numpy as np

def ibis_equidistant(input_str, Nwave, wrange=None):
    """
    Python equivalent of IDL function `ibis_equidistant`.
    
    Parameters:
    - input_str: Object containing 'grid' (assumed to be an array).
    - Nwave: Desired length of the output wavelength scale (should be 21 or more).
    - wrange: Optional range [start, end] for wavelength selection (default None).
    
    Returns:
    - wlambda: Unique, sorted wavelength values.
    - wlambda_equi: Equidistant wavelength scale.
    """
    
    # Stampa informazioni sul grid
    print("Shape of grid:", input_str.grid.shape)
    print("Contents of grid:", input_str.grid)
    print("wrange:", wrange)
    
    grid_data = input_str.grid[0]  # Estrai il vero array di dati
    print("Shape of grid_data:", grid_data.shape)
    print("Contents of grid_data:", grid_data)


    wlambda = np.unique(np.ravel(grid_data))
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
        nwlambda = round((np.max(wlambda) - np.min(wlambda)) / 0.01)
    else:
        nwlambda = 0  # O un altro valore sensato

    wlambda_equi = np.linspace(np.min(wlambda), np.max(wlambda), nwlambda, dtype=np.float64)

    return wlambda,wlambda_equi
