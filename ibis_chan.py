import numpy as np
from parmin_ibis import parmin_ibis
from lpff_pol import lpff_pol
from scipy.interpolate import interp1d, make_interp_spline
from multiprocessing import Pool
from functools import partial

"""
def ibis_chan(aps1,flat1,wscale1,wscale_equi,Npol,Nwave,Nx,Ny,npoints):


    # Define aperture indices
    id= np.where(aps1 >= 1)  

    # Ottieni gli indici delle aperture per i canali sinistro e destro
    idx = np.column_stack(np.where(aps1 >= 1))
    idxy = idx[:, 0]
    idxx = idx[:, 1]
    Numl = idxx.size

    avgprof = np.zeros((Npol, Nwave), dtype=np.float64)

    # Compute average profiles
    for j in range(Npol):
        for i in range(Nwave):

            print(f"Processing: i={i}, j={j}")     
            avgprof[j,i] = np.mean(flat1[j, i, :, :][idxx])
               

        avgoffs = np.zeros(Npol, dtype=float)
       
        avgoffs1 = np.zeros(Npol, dtype=float)

    # Interpolate and compute polynomial fitting
    for j in range(Npol):

        pavg = interp1d(wscale1, avgprof[j, :], kind='quadratic', fill_value="extrapolate")(wscale_equi)

        avgoffs[j] = lpff_pol(pavg)  # Filtro passa-basso
        avgoffs1[j] = parmin_ibis(pavg, npoints)

    bshift_cog1 = np.zeros((Npol, Ny, Nx//2), dtype=float)
    bshift_poly1 = np.zeros((Npol, Ny, Nx//2), dtype=float)

    for j in range(Npol):
        for i in range(Numl):
            x, y = idxx[i], idxy[i]

            # Verifica che gli indici siano validi
            if x >= flat1.shape[3]:  
                x = flat1.shape[3] - 1  # Limita x all'ultimo indice valido
            if y >= flat1.shape[2]:  
                y = flat1.shape[2] - 1  # Limita y all'ultimo indice valido

                    
            p = interp1d(wscale1, flat1[j, :, y, x], kind='quadratic', fill_value="extrapolate")(wscale_equi)
            tmp = lpff_pol(p)
        
            bshift_cog1[j, y, x] = tmp - avgoffs[j]
            bshift_poly1[j, y, x] = parmin_ibis(p, npoints) - avgoffs1[j]

    
    if Npol == 1:
        print("1")
        bshift_cog1 = bshift_cog1
        bshift_poly1= bshift_poly1

    else:
        print("2")
        bshift_cog1 = np.mean(bshift_cog1, axis=0)
        bshift_poly1 = np.mean(bshift_poly1, axis=0)


    return bshift_cog1,bshift_poly1
"""

def avgoffset(k, aps1, flat1, wscale1, wscale_equi, npoints, idxx, Nwave):
    # just for a check, not used in the final calculation
    id = np.where(aps1 >= 1, True, False)
    avgprof = np.mean(flat1[k, :, :, :], axis=(1, 2), where=id, dtype=np.float64)
    pavg = (make_interp_spline(wscale1, avgprof, k=2))(wscale_equi)
    print("avgprof", avgprof)
    print("pavg", pavg)
    avgoffs, avgoffs1 = lpff_pol(pavg), parmin_ibis(pavg, npoints)  # Filtro passa-basso
    print("Done", k)
    return [avgoffs, avgoffs1]

def avgoffsets(k, aps1, flat1, wscale1, wscale_equi, npoints):
    # This is the function being used in the final calculation
    id = np.where(aps1 >= 1, True, False)
    avgprof = np.mean(flat1[k, :, :, :], axis=(1, 2), where=id, dtype=float)
    pavg = (make_interp_spline(wscale1, avgprof, k=2))(wscale_equi)
    avgoffs, avgoffs1 = lpff_pol(pavg), parmin_ibis(pavg, npoints)  # Filtro passa-basso
    return [avgoffs, avgoffs1]

def interpolate(k, aps1, flat1, wscale1, wscale_equi, npoints, Nx, Ny, idxx, idxy):
    avgoffs, avgoffs1 = avgoffsets(k, aps1, flat1, wscale1, wscale_equi, npoints)
    bshift_cog1, bshift_poly1 = np.zeros((Ny, Nx//2), dtype=np.float64), np.zeros((Ny, Nx//2), dtype=np.float64)
    for i in range(idxx.size):
        x, y = idxx[i], idxy[i]
        p = (make_interp_spline(wscale1, flat1[k, :, y, x], k=2))(wscale_equi)
        bshift_cog1[y, x], bshift_poly1[y, x] = (lpff_pol(p) - avgoffs), (parmin_ibis(p, npoints) - avgoffs1)
    print("Done", k)
    return [bshift_cog1, bshift_poly1]
    
def ibis_chan(aps1, flat1, wscale1, wscale_equi, Npol, Nwave, Nx, Ny, npoints):

    # Ottieni gli indici delle aperture per i canali sinistro e destro
    idx = np.column_stack(np.where(aps1 >= 1))
    idxy, idxx = idx[:, 0], idx[:, 1]
# %%
    """
    # This part is not strictly necessary in the running code; it is included here for verification purposes.
    part = partial(avgoffset, aps1=aps1, flat1=flat1, wscale1=wscale1,
                   wscale_equi=wscale_equi, npoints=npoints, idxx=idxx, Nwave=Nwave)
    with Pool(6) as pool:
        result = pool.map(part, ([0, 1, 2, 3, 4, 5]))

    result = np.array(result)
    print(result.shape)
    avgoffs, avgoffs1 = result[:, 0], result[:, 1]
    print("avgoffs, avgoffs1 = ", avgoffs, avgoffs1)
    """
# %%
    
    # Interpolate and compute polynomial fitting
    part_interp = partial(interpolate, aps1=aps1, flat1=flat1, wscale1=wscale1,
                    wscale_equi=wscale_equi, npoints=npoints, Nx=Nx, Ny=Ny, idxx=idxx, idxy=idxy)
    # partial definition of the interpolation function.
    with Pool(6) as pool:
        result = pool.map(part_interp, ([0, 1, 2, 3, 4, 5]))
    # Calculations for all polarisation states are done in parallel.

    result = np.array(result)

    bshift_cog1, bshift_poly1 = result[:, 0, :, :], result[:, 1, :, :]

    if Npol == 1:
        print("1")
        bshift_cog1, bshift_poly1 = bshift_cog1, bshift_poly1

    else:
        print("2")
        bshift_cog1, bshift_poly1 = np.mean(bshift_cog1, axis=0), np.mean(bshift_poly1, axis=0)

    return bshift_cog1, bshift_poly1
