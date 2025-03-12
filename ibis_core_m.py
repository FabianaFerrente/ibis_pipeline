import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from ibis_equidistant import ibis_equidistant
from parmin_ibis import parmin_ibis
from lpff_pol import lpff_pol
import matplotlib.pyplot as plt
import os


def ibis_core_m(flat, wrange, npoints, aps, info_str, POL, DUAL, SINGLE, stokes=1):


    """
    Python conversion of IDL function `ibis_core_map`

    Parameters:
    - flat: 4D numpy array (Nx, Ny, Nwave, Npol)
    - wrange: Wavelength range
    - npoints: Number of points for calculations
    - aps: Aperture mask
    - info_str: Metadata information
    - pol: Flag for polarization mode
    - single: Single-beam mode flag
    - dual: Dual-beam mode flag
    - stokes: Number of polarization states (if any)

    1. Get dimensions
    2. Create output structure
    3. Spectropolarimetric Mode
    4. Dual-Beam Mode
    5. Compute Offsets
       Calcola la media sugli stati di polarizzazione
    6. Surface Fit
       Creazione di matrici x e y
       Fit left region
       Fit right region
       Creazione della mappa di offset

    Returns:
    - offset_map: Dictionary with computed offset maps
    """

    # ********************
    # 1. Get dimensions
    # ********************
    Nx, Ny, Nwave = flat.shape[:3]
    Npol = stokes if stokes is not None else 1

    # ********************
    # 2. Create output structure
    # ********************
    offset_map = {
        'cog': np.zeros((Nx, Ny), dtype=np.float64),
        'cog_fit': np.zeros((Nx, Ny), dtype=np.float64),
        'poly': np.zeros((Nx, Ny), dtype=np.float64),
        'poly_fit': np.zeros((Nx, Ny), dtype=np.float64)
    }

    # ********************
    # 3. Spectropolarimetric Mode
    # ********************
    if POL is not None and POL == 1:
        print("Spectropolarimetric mode activated.")
   
    wscale,wscale_equi = ibis_equidistant(info_str, Nwave, wrange)

    wscale1 = flat[0, 0, :, 0]
    # ********************
    # 4. Dual-Beam Mode
    # ********************
    if DUAL is not None and DUAL == 1:
        print("Dual-beam mode activated.")

        # Define left and right aperture indices
        idxxl = np.where(aps[:Nx//2, :] >= 1)
        idxxr = np.where(aps[Nx//2:, :] >= 1)

        # Ottieni gli indici delle aperture per i canali sinistro e destro
        idxl = np.array(np.where(aps[:Nx//2, :] >= 1)).T
        idxlx = idxl[:, 0]
        idxly = idxl[:, 1]
        Numl = len(idxlx)

        idxr = np.array(np.where(aps[Nx//2:, :] >= 1)).T
        idxrx = idxr[:, 0]
        idxry = idxr[:, 1]
        Numr = len(idxrx)

        avgprof_l = np.zeros((Nwave, Npol), dtype=np.float64)
        avgprof_r = np.zeros((Nwave, Npol), dtype=np.float64)
        print(avgprof_l.shape,avgprof_r.shape)


        # Compute average profiles
        for j in range(Npol):
            for i in range(Nwave):
                avgprof_l[i, j] = np.mean(flat[..., i, j][idxxl])
                avgprof_r[i, j] = np.mean(flat[..., i, j][idxxr])

        avgoffs_l = np.zeros(Npol)
        avgoffs_r = np.zeros(Npol)
        avgoffs1_l = np.zeros(Npol)
        avgoffs1_r = np.zeros(Npol)

        print(avgprof_l.shape,avgprof_r.shape)
        print("Dimensione wscale:", len(wscale1))
        print(wscale1)
        print("Dimensione avgprof_l[:, j]:", avgprof_l[:, j].shape)
        print("Dimensione wscale_equi:", len(wscale_equi))

        # Interpolate and compute polynomial fitting
        for j in range(Npol):
   
            pavg_l = interp1d(wscale1, avgprof_l[:, j], kind='quadratic', fill_value="extrapolate")(wscale_equi)
            tmp_l = lpff_pol(pavg_l)  # Filtro passa-basso
            avgoffs_l[j] = tmp_l
            avgoffs1_l[j] = parmin_ibis(pavg_l, npoints)

        # Interpolazione equidistante per il profilo destro
            pavg_r = interp1d(wscale1, avgprof_r[:, j], kind='quadratic', fill_value="extrapolate")(wscale_equi)
            tmp_r = lpff_pol(pavg_r)  # Filtro passa-basso
            avgoffs_r[j] = tmp_r
            avgoffs1_r[j] = parmin_ibis(pavg_r, npoints)
      

        # ********************
        # 5. Compute Offsets
        # ********************        
        print("5. Compute Offsets.")

#*************************************************************************
        # Parte momentanea per non rifare il ciclo
        save_file = "ciclo_output.npz"
        dati_caricati = False  # Flag per capire se abbiamo caricato i dati

        if os.path.exists(save_file):
            try:
        # Carica i dati salvati
                data = np.load(save_file)
                bshift_cog_l = data['bshift_cog_l']
                bshift_poly_l = data['bshift_poly_l']
                bshift_cog_r = data['bshift_cog_r']
                bshift_poly_r = data['bshift_poly_r']
                print("Dati caricati da file, cicli saltati.")
                dati_caricati = True
            except Exception as e:
                print(f"Errore nel caricamento del file {save_file}, eseguo i cicli. Errore: {e}")

#*************************************************************************

        if not dati_caricati:  # Esegui solo se i dati non sono stati caricati
            print("Eseguo i cicli e salvo i risultati...")
    
            bshift_cog_l = np.zeros((Nx//2, Ny, Npol))
            bshift_poly_l = np.zeros((Nx//2, Ny, Npol))
            bshift_cog_r = np.zeros((Nx//2, Ny, Npol))
            bshift_poly_r = np.zeros((Nx//2, Ny, Npol))
    
            flat_l = flat[0:Nx//2, :, :, :]  # Usa // per la divisione intera
            flat_r = flat[Nx//2:Nx, :, :, :]

            for j in range(Npol):
                for i in range(len(idxxl[0])):
                    x, y = idxxl[0][i], idxxl[1][i]
                    p = interp1d(wscale1, flat_l[x, y, :, j], kind='quadratic', fill_value="extrapolate")(wscale_equi)
                    tmp = lpff_pol(p)  # Applica la funzione lpff_pol
                    bshift_cog_l[idxlx[i], idxly[i], j] = tmp - avgoffs_l[j]
                    bshift_poly_l[idxlx[i], idxly[i], j] = parmin_ibis(p, npoints) - avgoffs1_l[j]
                    print("i.", i)
                for i in range(len(idxxr[0])):
                    x, y = idxxr[0][i], idxxr[1][i]
                    p = interp1d(wscale1, flat_r[x, y, :, j], kind='quadratic', fill_value="extrapolate")(wscale_equi)
                    tmp = lpff_pol(p)
                    bshift_cog_r[idxrx[i], idxry[i], j] = tmp - avgoffs_r[j]
                    bshift_poly_r[idxrx[i], idxry[i], j] = parmin_ibis(p, npoints) - avgoffs1_r[j]

#*************************************************************************
# Salva i risultati solo se sono stati calcolati
            np.savez(save_file, 
             bshift_cog_l=bshift_cog_l, 
             bshift_poly_l=bshift_poly_l, 
             bshift_cog_r=bshift_cog_r, 
             bshift_poly_r=bshift_poly_r)
            print("Risultati salvati in", save_file)

#*************************************************************************

        print('1',bshift_cog_l.shape,bshift_cog_r.shape)
        print('1',bshift_poly_l.shape,bshift_poly_r.shape)
        # Calcola la media sugli stati di polarizzazione
       
        if Npol == 1:
            print("1")
            bshift_cog_l = bshift_cog_l
            bshift_poly_l = bshift_poly_l

            bshift_cog_r = bshift_cog_r
            bshift_poly_r = bshift_poly_r
        else:
            print("2")
            bshift_cog_l = np.mean(bshift_cog_l, axis=2)
            bshift_poly_l = np.mean(bshift_poly_l, axis=2)

            bshift_cog_r = np.mean(bshift_cog_r, axis=2)
            bshift_poly_r = np.mean(bshift_poly_r, axis=2)
       

        print(bshift_cog_l.shape,bshift_cog_r.shape)
        print(bshift_poly_l.shape,bshift_poly_r.shape)
        # Assemblaggio dei risultati
        bshift_cog = np.zeros((Nx, Ny), dtype=float)
        bshift_poly = np.zeros((Nx, Ny), dtype=float)

        print(bshift_cog.shape,bshift_poly.shape)

        bshift_cog[0:Nx//2, :] = bshift_cog_l
        bshift_poly[0:Nx//2, :] = bshift_poly_l
        bshift_cog[Nx//2:Nx, :] = bshift_cog_r
        bshift_poly[Nx//2:Nx, :] = bshift_poly_r

        # ********************
        # 6. Surface Fit
        # ********************

        bshift_cog_fit = np.zeros((Nx, Ny))
        bshift_poly_fit = np.zeros((Nx, Ny))

        bshift_cog_fit_left = np.zeros((Nx//2, Ny))
        bshift_poly_fit_left = np.zeros((Nx//2, Ny))
        bshift_cog_fit_right = np.zeros((Nx//2, Ny))
        bshift_poly_fit_right = np.zeros((Nx//2, Ny))


        # Creazione di matrici x e y (equivalente di findgen + rebin)
    
        xmatrix, ymatrix = np.meshgrid(np.arange(Nx//2), np.arange(Ny), indexing='ij')
        # Stampa per verifica
        print("xmatrix shape:", xmatrix.shape)
        print("ymatrix shape:", ymatrix.shape)

        # Fit left region
        zmatrix_cog = bshift_cog_l
        zmatrix_poly = bshift_poly_l
        
        mask = zmatrix_cog != zmatrix_cog[10, 10]  # Maschera per valori validi
        points = np.column_stack((xmatrix[mask], ymatrix[mask]))

        fit_cog_left = griddata(points, zmatrix_cog[mask], (xmatrix, ymatrix), method='cubic')
        fit_poly_left = griddata(points, zmatrix_poly[mask], (xmatrix, ymatrix), method='cubic')

        bshift_cog_fit_left = np.nan_to_num(fit_cog_left)
        bshift_poly_fit_left = np.nan_to_num(fit_poly_left)

        # Fit right region
        zmatrix_cog = bshift_cog_r
        zmatrix_poly = bshift_poly_r

        mask = zmatrix_cog != zmatrix_cog[10, 10]
        points = np.column_stack((xmatrix[mask], ymatrix[mask]))

        fit_cog_right = griddata(points, zmatrix_cog[mask], (xmatrix, ymatrix), method='cubic')
        fit_poly_right = griddata(points, zmatrix_poly[mask], (xmatrix, ymatrix), method='cubic')

        bshift_cog_fit_right = np.nan_to_num(fit_cog_right)
        bshift_poly_fit_right = np.nan_to_num(fit_poly_right)

        # Assemblaggio finale
        bshift_cog_fit[:Nx//2, :] = bshift_cog_fit_left
        bshift_poly_fit[:Nx//2, :] = bshift_poly_fit_left
        bshift_cog_fit[Nx//2:, :] = bshift_cog_fit_right
        bshift_poly_fit[Nx//2:, :] = bshift_poly_fit_right

        # Creazione della mappa di offset
        offset_map = {
            "cog": bshift_cog_fit * 0.01,
            "cog_fit": bshift_cog_fit * 0.01,
            "poly": bshift_poly_fit * 0.01,
            "poly_fit": bshift_poly_fit * 0.01
        }

        # Visualizza la mappa 'cog_fit'
        plt.imshow(offset_map['cog_fit'], cmap='gray', aspect='auto')
        plt.colorbar()
        plt.title("Offset Map (cog_fit)")
        plt.show()

    return offset_map
