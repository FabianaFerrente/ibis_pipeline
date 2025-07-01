import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from ibis_equidistant import ibis_equidistant
from parmin_ibis import parmin_ibis
from lpff_pol import lpff_pol
import matplotlib.pyplot as plt
import os
import estrai_key as ek
from ibis_chan import ibis_chan

key_pol = 'pol'  # La chiave per estrarre il valore pol
key_dual = 'dual'  # La chiave per estrarre il valore dual
key_single = 'single'  # La chiave per estrarre il valore single
key_npoints = 'npoints'  # La chiave per estrarre il valore npoints

pol=ek.estrai_key('ibis_config_20150518.dat',key_pol)
dual=ek.estrai_key('ibis_config_20150518.dat',key_dual)
single=ek.estrai_key('ibis_config_20150518.dat',key_single)
npoints=ek.estrai_key('ibis_config_20150518.dat',key_npoints)

pol = int(pol)  
dual = int(dual)
single = int(single)
npoints = int(points)

def surface_fit(bshift_cog, bshift_poly, Ny, Sep_LR):
    xmatrix, ymatrix = np.tile(np.arange(Sep_LR), (Ny, 1)), np.tile(np.arange(Ny).reshape(Ny, 1), (1, Sep_LR))
    zmatrix_cog, zmatrix_poly = bshift_cog, bshift_poly

    mask = zmatrix_cog != zmatrix_cog[10, 10]

    x_valid, y_valid = xmatrix[mask].flatten(), ymatrix[mask].flatten()
    z_cog_valid, z_poly_valid = zmatrix_cog[mask].flatten(), zmatrix_poly[mask].flatten()

    # Regressione polinomiale (grado 2)
    X = np.vstack((x_valid, y_valid)).T
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model_cog, model_poly = LinearRegression(), LinearRegression()
    model_cog.fit(X_poly, z_cog_valid)
    model_poly.fit(X_poly, z_poly_valid)
    fit_cog, fit_poly = model_cog.predict(X_poly), model_poly.predict(X_poly)

    # Matrici risultato (inizializzate con NaN)
    bshift_cog_fit = np.full_like(zmatrix_cog, np.nan, dtype=np.float64)
    bshift_poly_fit = np.full_like(zmatrix_poly, np.nan, dtype=np.float64)

    # Inserimento solo nei punti originali (come in IDL)
    bshift_cog_fit[mask] = fit_cog
    bshift_poly_fit[mask] = fit_poly

    return bshift_cog_fit, bshift_poly_fit



def ibis_core_m(flat, wrange, aps, info_str, stokes=1):


    """
    Python conversion of IDL function `ibis_core_map`

    Parameters:
    - flat: 4D numpy array (Npol, Nwave, Ny, Nx)
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
    print("Shape of flat:", flat.shape)
    Nwave, Ny, Nx = flat.shape[1:4]
    Npol = stokes if stokes is not None else 1
    Sep_LR = Nx//2  # separation index between the channels, start index of the second channel
    print("Nwave:", Nwave)
    # ********************
    # 2. Create output structure
    # ********************
    offset_map = {
        'cog': np.zeros((Ny, Nx), dtype=np.float64),
        'cog_fit': np.zeros((Ny, Nx), dtype=np.float64),
        'poly': np.zeros((Ny, Nx), dtype=np.float64),
        'poly_fit': np.zeros((Ny, Nx), dtype=np.float64)
    }

    # ********************
    # 3. Spectropolarimetric Mode
    # ********************
    if pol is not None and pol == 1:
        print("Spectropolarimetric mode activated.")
   
        wscale1,wscale_equi = ibis_equidistant(info_str, Nwave, wrange)
    

    # ********************
    # 4. Dual-Beam Mode
    # ********************
    if dual is not None and dual == 1:
        
        print("Dual-beam mode activated.")
        flat_l, apsl = flat[:, :, :, :Sep_LR], aps[:, :Sep_LR]
        flat_r, apsr = flat[:, :, :, Sep_LR:], aps[:, Sep_LR:]

        # Defining arrays to store final results
        bshift_cog, bshift_poly = np.zeros((Ny, Nx), dtype=float), np.zeros((Ny, Nx), dtype=float)
        bshift_cog_fit, bshift_poly_fit = np.zeros((Ny, Nx), dtype=float), np.zeros((Ny, Nx), dtype=float)

        # Defining arrays for storing the surface fits
        bshift_cog_fit_left, bshift_poly_fit_left = np.zeros((Ny, Sep_LR), dtype=float), np.zeros((Ny, Sep_LR), dtype=float)
        bshift_cog_fit_right, bshift_poly_fit_right = np.zeros((Ny, Sep_LR), dtype=float), np.zeros((Ny, Sep_LR), dtype=float)
        
        """
        flat_l=flat[:, :, :, :Nx//2]
        apsl = aps[:, 0:Nx//2]

        bshift_cog1,bshift_poly1=ibis_chan(apsl,flat_l,wscale1,wscale_equi,Npol,Nwave,Nx,Ny,npoints)
        # moved down

        bshift_cog_l = bshift_cog1
        bshift_poly_l = bshift_poly1

        flat_r=flat[:, :, :, Nx//2:]
        apsr = aps[:, Nx//2:Nx]

        bshift_cog1,bshift_poly1=ibis_chan(apsr,flat_r,wscale1,wscale_equi,Npol,Nwave,Nx,Ny,npoints)

        bshift_cog_r = bshift_cog1
        bshift_poly_r = bshift_poly1    
        """

        bshift_cog_l, bshift_poly_l = ibis_chan(apsl, flat_l, wscale1, wscale_equi, Npol, Nwave, Nx, Ny, npoints)
        bshift_cog_r, bshift_poly_r = ibis_chan(apsr, flat_r, wscale1, wscale_equi, Npol, Nwave, Nx, Ny, npoints)
        
        
        print(bshift_cog_l.shape,bshift_cog_r.shape)
        print(bshift_poly_l.shape,bshift_poly_r.shape)
        """
        # Assemblaggio dei risultati
        bshift_cog = np.zeros((Ny, Nx), dtype=float)
        bshift_poly = np.zeros((Ny, Nx), dtype=float)
        # moved up after defining flat_l
        """
        print(bshift_cog.shape,bshift_poly.shape)

        """
        bshift_cog[:, 0:Nx//2] = bshift_cog_l
        bshift_poly[:, 0:Nx//2] = bshift_poly_l
        bshift_cog[:, Nx//2:Nx] = bshift_cog_r
        bshift_poly[:, Nx//2:Nx] = bshift_poly_r
        """
        bshift_cog[:, :Sep_LR], bshift_poly[:, :Sep_LR] = bshift_cog_l, bshift_poly_l # left channel
        bshift_cog[:, Sep_LR:], bshift_poly[:, Sep_LR:] = bshift_cog_r, bshift_poly_r # right channel

        

        # Visualizza la mappa 'cog_fit'
        plt.figure(figsize=(7, 4))
        plt.imshow(bshift_cog, cmap='gray', aspect='auto')
        plt.colorbar()
        plt.title("(cog)")
        # plt.title("(cog_fit)")
        plt.show()

        # ********************
        # 6. Surface Fit
        # ********************

        """
        bshift_cog_fit = np.zeros((Ny, Nx))
        bshift_poly_fit = np.zeros((Ny, Nx))

        bshift_cog_fit_left = np.zeros((Ny, Nx//2))
        bshift_poly_fit_left = np.zeros((Ny, Nx//2))
        bshift_cog_fit_right = np.zeros((Ny, Nx//2))
        bshift_poly_fit_right = np.zeros((Ny, Nx//2))
        # moved both up near the start of the if condition.

        # Creazione di matrici x e y
        # Coordinate come in IDL con rebin
        xmatrix = np.tile(np.arange(Nx//2), (Ny, 1))
        ymatrix = np.tile(np.arange(Ny).reshape(Ny, 1), (1, Nx//2))
        
        zmatrix_cog = bshift_cog_l
        zmatrix_poly = bshift_poly_l
        
        # Maschera dei punti validi
        mask = zmatrix_cog != zmatrix_cog[10, 10]
        x_valid = xmatrix[mask].flatten()
        y_valid = ymatrix[mask].flatten()
        z_cog_valid = zmatrix_cog[mask].flatten()
        z_poly_valid = zmatrix_poly[mask].flatten()
        
        # Regressione polinomiale (grado 2)
        X = np.vstack((x_valid, y_valid)).T
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        # Fit per zmatrix_cog
        model_cog = LinearRegression()
        model_cog.fit(X_poly, z_cog_valid)
        fit_cog = model_cog.predict(X_poly)

        # Fit per zmatrix_poly
        model_poly = LinearRegression()
        model_poly.fit(X_poly, z_poly_valid)
        fit_poly = model_poly.predict(X_poly)
        
        # Matrici risultato (inizializzate con NaN)
        bshift_cog_fit_left = np.full_like(zmatrix_cog, np.nan, dtype=np.float64)
        bshift_poly_fit_left = np.full_like(zmatrix_poly, np.nan, dtype=np.float64)

        # Inserimento solo nei punti originali (come in IDL)
        bshift_cog_fit_left[mask] = fit_cog
        bshift_poly_fit_left[mask] = fit_poly
    
        
        # Fit the right region
        xmatrix = np.tile(np.arange(Nx//2, Nx), (Ny, 1))
        ymatrix = np.tile(np.arange(Ny).reshape(Ny, 1), (1, Nx//2))
        
        # Matrici da interpolare
        zmatrix_cog = bshift_cog_r
        zmatrix_poly = bshift_poly_r

        # Maschera dei valori validi
        mask = zmatrix_cog != zmatrix_cog[10, 10]
        x_valid = xmatrix[mask].flatten()
        y_valid = ymatrix[mask].flatten()
        z_cog_valid = zmatrix_cog[mask].flatten()
        z_poly_valid = zmatrix_poly[mask].flatten()

        # Fit polinomiale di secondo grado
        X = np.vstack((x_valid, y_valid)).T
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        # Fit su zmatrix_cog
        model_cog = LinearRegression()
        model_cog.fit(X_poly, z_cog_valid)
        fit_cog = model_cog.predict(X_poly)

        # Fit su zmatrix_poly
        model_poly = LinearRegression()
        model_poly.fit(X_poly, z_poly_valid)
        fit_poly = model_poly.predict(X_poly)

        # Matrici risultato inizializzate con NaN
        bshift_cog_fit_right = np.full_like(zmatrix_cog, np.nan, dtype=np.float64)
        bshift_poly_fit_right = np.full_like(zmatrix_poly, np.nan, dtype=np.float64)

        # Inserimento dei valori fitted nei punti originali
        bshift_cog_fit_right[mask] = fit_cog
        bshift_poly_fit_right[mask] = fit_poly
        """

        # Assemblaggio finale
        """
        bshift_cog_fit[:, :Nx//2] = bshift_cog_fit_left
        bshift_poly_fit[:, :Nx//2] = bshift_poly_fit_left
        bshift_cog_fit[:, Nx//2:] = bshift_cog_fit_right
        bshift_poly_fit[:, Nx//2:] = bshift_poly_fit_right
        """
        bshift_cog_fit[:, :Sep_LR], bshift_poly_fit[:, :Sep_LR] = surface_fit(bshift_cog_l, bshift_poly_l, Ny, Sep_LR)
        bshift_cog_fit[:, Sep_LR:], bshift_poly_fit[:, Sep_LR:] = surface_fit(bshift_cog_r, bshift_poly_r, Ny, Sep_LR)

        # Creazione della mappa di offset
        offset_map = {
            "cog": bshift_cog * 0.01,
            "cog_fit": bshift_cog_fit * 0.01,
            "poly": bshift_poly * 0.01,
            "poly_fit": bshift_poly_fit * 0.01
        }

        # Visualizza la mappa 'cog_fit'
        plt.figure(figsize=(7, 4))
        plt.imshow(offset_map['cog_fit'], cmap='gray', aspect='auto')
        plt.colorbar()
        plt.title("Offset Map (cog_fit)")
        plt.show()

    return offset_map
