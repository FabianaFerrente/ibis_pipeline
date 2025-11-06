import numpy as np
from parmin_ibis import parmin_ibis
from lpff_pol import lpff_pol
from scipy.interpolate import make_interp_spline
from multiprocessing import Pool
from functools import partial

def avgoffsets(k, aps1, flat1, wscale1, wscale_equi, npoints, is_spectroscopic):
    # mask 2D (Ny, Nx)
    mask_flat = (aps1 >= 1).ravel()

    if is_spectroscopic:
        # flat1 può essere (Nwave, Ny, Nx) oppure (1, Nwave, Ny, Nx) se è stato aggiunto l'asse polare
        f = flat1 if flat1.ndim == 3 else flat1[0]  # -> (Nwave, Ny, Nx)
        Nwave = f.shape[0]
        flat_reshaped = f.reshape(Nwave, -1)        # -> (Nwave, Ny*Nx)
        if mask_flat.size != flat_reshaped.shape[1]:
            raise ValueError(f"Mask size {mask_flat.size} != spatial size {flat_reshaped.shape[1]}")
        masked_data = flat_reshaped[:, mask_flat]   # -> (Nwave, N_valid)
        if masked_data.shape[1] == 0:
            # nessun pixel valido: profilo medio a zero
            avgprof = np.zeros(Nwave, dtype=float)
        else:
            avgprof = np.mean(masked_data, axis=1, dtype=float)
    else:
        # flat1: (Npol, Nwave, Ny, Nx)
        f = flat1[k]                                 # -> (Nwave, Ny, Nx)
        Nwave = f.shape[0]
        flat_reshaped = f.reshape(Nwave, -1)         # -> (Nwave, Ny*Nx)
        if mask_flat.size != flat_reshaped.shape[1]:
            raise ValueError(f"Mask size {mask_flat.size} != spatial size {flat_reshaped.shape[1]}")
        masked_data = flat_reshaped[:, mask_flat]    # -> (Nwave, N_valid)
        if masked_data.shape[1] == 0:
            avgprof = np.zeros(Nwave, dtype=float)
        else:
            avgprof = np.mean(masked_data, axis=1, dtype=float)

    # (opzionale ma prudente) rimuove eventuali NaN/Inf residui
    avgprof = np.nan_to_num(avgprof, nan=0.0, posinf=0.0, neginf=0.0)

    pavg = make_interp_spline(wscale1, avgprof, k=2)(wscale_equi)
    avgoffs = lpff_pol(pavg)
    avgoffs1 = parmin_ibis(pavg, npoints)
    return avgoffs, avgoffs1

def interpolate(k, aps1, flat1, wscale1, wscale_equi, npoints, Nx, Ny, idxx, idxy, is_spectroscopic):
    avgoffs, avgoffs1 = avgoffsets(k, aps1, flat1, wscale1, wscale_equi, npoints, is_spectroscopic)

    bshift_cog1 = np.zeros((Ny, Nx//2), dtype=np.float64)
    bshift_poly1 = np.zeros((Ny, Nx//2), dtype=np.float64)

    for i in range(idxx.size):
        x, y = int(idxx[i]), int(idxy[i])
        if is_spectroscopic:
            # flat1 può essere (Nwave, Ny, Nx) oppure (1, Nwave, Ny, Nx)
            if flat1.ndim == 3:
                profile = flat1[:, y, x]
            else:
                profile = flat1[0, :, y, x]
        else:
            profile = flat1[k, :, y, x]

        p = make_interp_spline(wscale1, profile, k=2)(wscale_equi)
        bshift_cog1[y, x] = lpff_pol(p) - avgoffs
        bshift_poly1[y, x] = parmin_ibis(p, npoints) - avgoffs1

    print("Done", k)
    return [bshift_cog1, bshift_poly1]

def ibis_chan(aps1, flat1, wscale1, wscale_equi, Nwave, Npol, Nx, Ny, npoints):
    is_spectroscopic = (flat1.ndim == 3)  # True se spettroscopico, False se spettropolarimetrico

    idx = np.column_stack(np.where(aps1 >= 1))
    idxy, idxx = idx[:, 0], idx[:, 1]

    if is_spectroscopic:
        Npol = 1
        flat1 = flat1[np.newaxis, ...]  # aggiunge asse polare per uniformità

    part_interp = partial(
        interpolate,
        aps1=aps1,
        flat1=flat1,
        wscale1=wscale1,
        wscale_equi=wscale_equi,
        npoints=npoints,
        Nx=Nx,
        Ny=Ny,
        idxx=idxx,
        idxy=idxy,
        is_spectroscopic=is_spectroscopic
    )

    # numero reale di polarizzazioni da processare
    num_pol = flat1.shape[0]

    with Pool(min(num_pol, 6)) as pool:
        result = pool.map(part_interp, list(range(num_pol)))


    result = np.array(result)  # shape: (Npol, 2, Ny, Nx//2)
    bshift_cog1 = result[:, 0, :, :]
    bshift_poly1 = result[:, 1, :, :]

    if Npol == 1:
        bshift_cog1 = bshift_cog1[0]
        bshift_poly1 = bshift_poly1[0]
        print("Spectroscopic mode: single polarization")
    else:
        bshift_cog1 = np.mean(bshift_cog1, axis=0)
        bshift_poly1 = np.mean(bshift_poly1, axis=0)
        print("Polarimetric mode: averaged over polarizations")

    return bshift_cog1, bshift_poly1
