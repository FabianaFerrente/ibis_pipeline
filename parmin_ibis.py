import numpy as np

def parmin_ibis(y, npoints):
    s = y.shape
    yy = y.copy()  # Copia per evitare modifiche non volute all'array originale
    cut = 1
    xxpos = np.argmin(yy)  # Trova l'indice del minimo iniziale

    # Controllo per xxpos (assicurarsi che non esca dai limiti)
    while xxpos > (s[0] - npoints):
        sub_array = yy[cut * 10 : s[0] - cut * 10]
        if len(sub_array) == 0:
            break  # Evita problemi se l'intervallo diventa vuoto
        pmin_idx = np.argmin(sub_array)  # Trova il minimo nell'intervallo
        xxpos = pmin_idx + cut * 10  # Aggiorna xxpos con l'indice relativo all'intervallo
        cut += 1

    # Estrai la porzione di y per il fitting, verificando che l'intervallo sia valido
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
    if coef[0] != 0:  # Evita divisioni per zero
        zent = -coef[1] / (2 * coef[0])  # Calcolo del minimo
    else:
        zent = 0  # Se il polinomio è lineare, non possiamo fare il minimo

    # Restituisci la posizione minima
    xmin = zent + (xxpos - npoints)
    return xmin
