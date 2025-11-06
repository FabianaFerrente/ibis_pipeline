import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import readsav
from astropy.io import fits
from ibis_core_m import ibis_core_m  # Importa la funzione dal tuo file
import estrai_key as ek
import os

# ***************************************************************
# Configurazione
# ***************************************************************

config_file = 'ibis_config_20150518.dat'
filtro =  '6173'#'8542'  # o '6563'

# Specifica le chiavi da cercare nel file per le variabili necessarie

key_date = 'date'  # La chiave per estrarre la data
key_time = 'time_obs'  # La chiave per estrarre l'orario delle osservazioni
#key_pol = 'pol'  # La chiave per estrarre il valore pol
key_dual = 'dual'  # La chiave per estrarre il valore dual
key_single = 'single'  # La chiave per estrarre il valore single
key_npoints = 'npoints'  # La chiave per estrarre il valore npoints

date=ek.estrai_key(config_file,key_date)
time_obs=ek.estrai_key(config_file,key_time)
#pol=ek.estrai_key(config_file,key_pol)
dual=ek.estrai_key(config_file,key_dual)
single=ek.estrai_key(config_file,key_single)
npoints=ek.estrai_key(config_file,key_npoints)

dual = int(dual)
single = int(single)
npoints = int(npoints)

# Imposto pol in base al filtro
filtro = str(filtro)

if filtro in ["6173", "8542"]:
    pol = 1
elif filtro == "6563":
    pol = 0
else:
    raise ValueError(f"Filtro non riconosciuto: {filtro}")


# Ora associo il numero di polarizzazioni
if pol == 1:
    Npol = 6
    stokes = 6
else:
    Npol = 1
    stokes = 1

print(f"Filtro: {filtro} Å -> pol={pol}, Npol={Npol}")


print(f"Data: {date} - Time: {time_obs} - Filtro: {filtro}")
print(f"Pol: {pol}, Dual: {dual}, Single: {single}, Points: {npoints}, Stokes: {stokes}")


# *****************************************************************
# Caricamento e lettura dei file di flat,dark,configurazione e aps
# *****************************************************************

suffix = "_0" if filtro == '6174' else ""
flat_file = f"{filtro}_flat{suffix}.sav"
dark_file = f"{filtro}_dark{suffix}.sav"
aps_file = f"{filtro}_aps{suffix}.fits"


# Carica flat e dark e aps
for f in [flat_file, dark_file, aps_file]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File mancante: {f}")

flat_data, dark_data = readsav(flat_file), readsav(dark_file)
with fits.open(aps_file) as hdul:
    aps, aps_h = hdul[0].data, hdul[0].header

# I dati siano in variabili chiamate 'flat' e 'dark'
f_file, d_file = flat_data['flat'], dark_data['dark']
info_str = flat_data['info_flat_nb'] 

# ***************************************************************
# Ottieni le dimensioni
# ***************************************************************
print("Forma di f_file:", f_file.shape)
print("Forma di d_file:", d_file.shape)

Nwave = int(f_file.shape[0] / Npol)

Ny, Nx = f_file.shape[1], f_file.shape[2]

print(Nwave)

# Definizione wrange 

#wrange = [1, Nwave - 1]

if filtro == '6563':
    wrange = [2, Nwave - 3]
elif filtro == '8542':
    wrange = [3, Nwave - 3]
elif filtro == '6173':
    wrange = [1, Nwave - 1]
else:
    raise ValueError("Filtro non riconosciuto. Usa 6563, 8542 o 6173.")

print(f"Wave range: {wrange[0]} -> {wrange[1] - 1}")


# ***************************************************************
# Creare il "dark" medio, nessuna dipendenza dalla lunghezza d'onda
# ***************************************************************
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
# Creazione flat_tmp in base alla modalità
# ***************************************************************
if pol == 1:
    print("Modalità spettropolarimetrica")
    flat_tmp = np.zeros((Npol, Nwave, Ny, Nx), dtype=np.float32)
    for i in range(Npol):
        for n in range(Nwave):
            flat_tmp[i, n, :, :] = f_file[n*Npol+i, :, :] - dark
else:
    print("Modalità spettroscopica")
    flat_tmp = np.zeros((Nwave, Ny, Nx), dtype=np.float32)
    for n in range(Nwave):
        flat_tmp[n, :, :] = f_file[n, :, :] - dark

# ***************************************************************
# Visualizzazione profili (sia pol=0 che pol=1)
# ***************************************************************
plt.figure(figsize=(7, 4))

if pol == 1:
    # Per pol=1, scegliamo ad esempio la prima polarizzazione
    plt.plot(flat_tmp[0, :, 500, 250], marker='o', markersize=0.5, label="Posizione (250,500)")
    plt.plot(flat_tmp[0, :, 125, 250], marker='x', markersize=0.5, label="Posizione (250,125)")
    plt.plot(flat_tmp[0, :, 900, 250], marker='s', markersize=0.5, label="Posizione (250,900)")
else:
    # Per pol=0
    plt.plot(flat_tmp[:, 500, 250], marker='o', markersize=0.5, label="Posizione (250,500)")
    plt.plot(flat_tmp[:, 125, 250], marker='x', markersize=0.5, label="Posizione (250,125)")
    plt.plot(flat_tmp[:, 900, 250], marker='s', markersize=0.5, label="Posizione (250,900)")

plt.legend(fontsize=6)
plt.show(block=False)
plt.pause(3)
plt.close()


# ***************************************************************
# Calcolo offset
# ***************************************************************
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print("Inizio calcolo offset...")

    if pol == 0:
        # spettroscopico: flat_tmp ha shape (Nwave, Ny, Nx)
        # aggiungo un asse pol fittizio -> (1, Nwave, Ny, Nx)
        data_slice = flat_tmp[wrange[0]:wrange[1], :, :]
        data_slice = data_slice[np.newaxis, :, :, :]
    else:
        # spettropolarimetrico: flat_tmp ha già shape (Npol, Nwave, Ny, Nx)
        data_slice = flat_tmp[:, wrange[0]:wrange[1], :, :]

    # (opzionale) stampa di controllo
    print("data_slice shape =", data_slice.shape)  # atteso: (Npol, Nwave_sel, Ny, Nx)

    offset = ibis_core_m(data_slice, wrange, aps, info_str=info_str, stokes=stokes)

    print("Offset calcolato correttamente.")

    # Salvataggio in file .npz con nome in base al filtro
    out_file = f"offsets_{filtro}_new.npz"
    np.savez(out_file,
             cog=offset['cog'],
             cog_fit=offset['cog_fit'],
             poly=offset['poly'],
             poly_fit=offset['poly_fit'])

    print(f"Offset salvato in '{out_file}'.")
    print("Codice eseguito correttamente.")
