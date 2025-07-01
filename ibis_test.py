import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import readsav
from astropy.io import fits
from ibis_core_m import ibis_core_m  # Importa la funzione dal tuo file
import estrai_key as ek


# Specifica le chiavi da cercare nel file per le variabili necessarie

key_date = 'date'  # La chiave per estrarre la data
key_time = 'time_obs'  # La chiave per estrarre l'orario delle osservazioni
key_pol = 'pol'  # La chiave per estrarre il valore pol
key_dual = 'dual'  # La chiave per estrarre il valore dual
key_single = 'single'  # La chiave per estrarre il valore single
key_npoints = 'npoints'  # La chiave per estrarre il valore npoints

date=ek.estrai_key('ibis_config_20150518.dat',key_date)
time_obs=ek.estrai_key('ibis_config_20150518.dat',key_time)
pol=ek.estrai_key('ibis_config_20150518.dat',key_pol)
dual=ek.estrai_key('ibis_config_20150518.dat',key_dual)
single=ek.estrai_key('ibis_config_20150518.dat',key_single)
npoints=ek.estrai_key('ibis_config_20150518.dat',key_npoints) 

pol = int(pol)  
dual = int(dual)
single = int(single)
npoints = int(npoints)

stokes = 6 
filtro = '6173' # decide later where this variable should be taken

#check, da togliere dopo
print("Data - Time - filtro:", date, time_obs, filtro)
print("pol:",pol)
print("dual:",dual)
print("single:",single)
print("points:",npoints)

# *****************************************************************
# Caricamento e lettura dei file di flat,dark,configurazione e aps
# *****************************************************************

# Carica flat e dark e aps
flat_data, dark_data = readsav('6173_flat.sav'), readsav('6173_dark.sav')

"""
with fits.open('6173_aps.fits') as hdul:
    hdul.info()  # Mostra le estensioni del file
    aps = hdul[0].data  # Legge i dati dell'estensione principale
    aps_h = hdul[0].header  # Legge l'header
    idx = np.where(aps > 1)
"""
with fits.open('6173_aps.fits') as hdul:
    hdul.info()  # Mostra le estensioni del file
    aps, aps_h = hdul[0].data, hdul[0].header
idx = np.where(aps > 1)

# I dati siano in variabili chiamate 'flat' e 'dark'
f_file, d_file = flat_data['flat'], dark_data['dark']
info_str = flat_data['info_flat_nb'] 

# ***************************************************************
# Ottieni le dimensioni
# ***************************************************************
print("Forma di f_file:", f_file.shape)
print("Forma di d_file:", d_file.shape)

Ny, Nx, Npol = f_file.shape[1], f_file.shape[2], 6
Nwave = int(f_file.shape[0] / Npol)

print(Nwave)


# ***************************************************************
# Creare il "dark" medio, nessuna dipendenza dalla lunghezza d'onda
# ***************************************************************
# dark = np.mean(d_file, axis=1)
dark = np.mean(d_file, axis=0)
print("Dim dark:", dark.shape)

# ***************************************************************
# Ordinare i flats 
# ***************************************************************
index = np.arange(Nwave)
ftmp = np.copy(f_file)

for i in range(Nwave):
    ftmp[i*Npol:(i+1)*Npol, :, :] = f_file[index[i]*Npol:(index[i]+1)*Npol, :, :]

f_file = ftmp
del ftmp  # Libera memoria

# ***************************************************************
# Modalità spettropolarimetrica
# ***************************************************************
if pol == 1:
    print('------------------------------------------------------')
    print('Blueshift calculation for spectropolarimetric case ...')
    print('------------------------------------------------------')

    # Verifica delle dimensioni
    print("Forma di f_file:", f_file.shape)  # Deve essere (144, 1000, 1000)
    print("Forma di dark:", dark.shape)  # Deve essere (1000, 1000)

    #temp=np.zeros((Nx,Ny), dtype=np.float32)
    temp = np.zeros((Ny,Nx), dtype=np.float32)
    flat_tmp = np.zeros((Npol, Nwave, Ny, Nx), dtype=np.float32)
    
    # Ciclo corretto per calcolare il blueshift
    
    for i in range(Npol):
        for n in range(Nwave):
            temp = f_file[n*Npol+i, :, :] - dark[:, :]  # Corretto indexing
            flat_tmp[i, n, :, :] = temp

    # Visualizzazione della parte dell'immagine
    plt.figure(figsize=(7, 4))
    plt.plot(flat_tmp[0,:,500, 250], marker='o', markersize=0.5, label="Posizione (250,500)")  #[250, 500, :, 0]
    plt.plot(flat_tmp[0,:, 125, 250], marker='x', markersize=0.5, label="Posizione (250,125)")  #[250, 125, :, 0]
    plt.plot(flat_tmp[0,:,900, 250], marker='s', markersize=0.5, label="Posizione (250,900)")  #[250, 900, :, 0]
    plt.legend(fontsize=6)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

     # Define the wavelength range
    print('Nwave=',Nwave)
    wrange = [2, Nwave - 1]
    print(wrange,wrange[0],wrange[1])
# #***************************************************************
# #Calcolo dell'offset con ibis_core_m
# #***************************************************************
    offset = ibis_core_m(flat_tmp[:, wrange[0]:wrange[1], :, :], wrange, aps, info_str=info_str, stokes=stokes)

    print("Offset calcolato correttamente.")

# Salva più array in un file .npz
    np.savez('offsets.npz', cog=offset['cog'], cog_fit=offset['cog_fit'], poly=offset['poly'], poly_fit=offset['poly_fit'])
    print("Offset salvato in 'offset.fits'.")


# ***************************************************************
# Fine codice
# ***************************************************************
print("Codice eseguito correttamente.")
