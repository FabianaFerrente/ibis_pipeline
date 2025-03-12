import numpy as np

def parmin_ibis(y, npoints):
    s = y.shape
    yy = y.copy()
    cut = 1
    xxpos = np.argmin(yy)  # Trova l'indice del minimo iniziale
    pmin = np.min(yy)      # Trova il valore minimo

    # Controllo per xxpos
    while xxpos > (s[0] - npoints):
        sub_array = yy[cut * 10 : s[0] - cut * 10]
        if len(sub_array) == 0:
            break  # Evita problemi se l'intervallo diventa vuoto
        pmin_idx = np.argmin(sub_array)  # Trova il minimo nell'intervallo
        pmin = np.min(sub_array)  # Valore minimo
        xxpos = pmin_idx + cut * 10  # Aggiorna xxpos con l'indice relativo all'intervallo
        cut += 1

    # Estrai la porzione di y per il fitting
    if xxpos - npoints < 0 or xxpos + npoints + 1 > s[0]:
        return xxpos  # Evita errori se l'intervallo esce dai limiti

    yn = yy[xxpos - npoints : xxpos + npoints + 1]
    xn = np.arange(len(yn))  # Genera array di indici

    # Verifica che ci siano abbastanza punti per il fitting
    if len(yn) < 3:
        return xxpos  # Restituisce xxpos senza fitting se ci sono meno di 3 punti

    # Fitting polinomiale di secondo grado
    coef = np.polyfit(xn, yn, 2)

    # Calcola il punto minimo
    if coef[2] != 0:
        zent = -1 * coef[1] / (2 * coef[2])
    else:
        zent = 0  # Evita divisioni per zero

    xmin = zent + (xxpos - npoints)

    return xmin
