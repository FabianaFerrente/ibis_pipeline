def parsing(linea, key):
    
    linea = linea.strip()
    if linea.startswith(';'):  # Ignora le righe commentate
        return None
    
    if key in linea and '=' in linea:  # Controlla se la chiave e '=' sono presenti nella riga
        valore = linea.split('=')[1].strip()  # Prende la parte dopo '=' e rimuove spazi
        valore = valore.split()[0]  # Prende solo il primo elemento (ignora commenti/testo extra)
        return valore.replace("'", "").replace('"', "")  # Rimuove eventuali virgolette
    return None  # Se la chiave non è trovata, restituisce None


def estrai_key(percorso_file, key):
   
    with open(percorso_file, 'r') as file:

        for linea in file:
            res=parsing(linea,key)
            if res:  # Se la riga contiene la chiave
                return res  
        return None  # se non trova nulla
