import numpy as np
from parmin_ibis import parmin_ibis
from lpff_pol import lpff_pol
from scipy.interpolate import interp1d

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

    
  
