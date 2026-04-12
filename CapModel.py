#
# PyFibers funge da "ponte" (in Python) per il potente motore di simulazione NEURON. Per generare un CAP
# a una distanza specifica, il flusso di lavoro del nostro codice seguirà questi passaggi:
# 1. Definizione dei modelli di fibra: Creeremo una popolazione di fibre afferenti (ad esempio, usando il modello
# MRG integrato in PyFibers per le fibre mieliniche) con diametri diversi.
# 2. Configurazione dello stimolo: Definiremo una sorgente di stimolazione (es. un elettrodo extracellulare a punto)
#  posizionata vicino al nervo.
# 3. Posizionamento dell'elettrodo di registrazione: Imposteremo le coordinate spaziali (x,y,z) del nostro
# elettrodo virtuale. La coordinata z (l'asse lungo il nervo) rappresenterà la nostra distanza definita.
# 4. Simulazione e Somma (SFAP → CAP): Simuleremo ogni fibra per ottenere il suo Single Fiber Action Potential (SFAP)
# registrato dall'elettrodo. Poiché PyFibers sfrutta NEURON per calcolare la cinetica dei canali ionici,
# il ritardo temporale (dispersione) e la forma dell'onda dovuti alla distanza saranno fisicamente accurati.
# Infine, sommeremo i segnali nel tempo.

import numpy as np
# Le importazioni seguono la struttura tipica di PyFibers
from pyfibers.models import MRGFiber
from pyfibers.stimulation import PointSource
from pyfibers.recording import ExtracellularElectrode
from pyfibers.simulation import Simulator

import numpy as np
import matplotlib.pyplot as plt


def simula_cap_afferente():
    """
    Simula un Compound Action Potential (CAP) usando PyFibers
    a una distanza definita dal punto di stimolo.
    """

    # 1. Parametri della popolazione afferente (diametri in micrometri)
    diametri_fibre = [5.7, 7.3, 8.5, 10.0, 11.5]

    # 2. Definizione delle coordinate geometriche (in micrometri)
    z_stimolo = 0.0  # Origine lungo il nervo (punto di stimolazione)
    z_registrazione = 50000.0  # Distanza definita: 50 mm (50000 um)
    distanza_radiale = 1000.0  # L'elettrodo dista 1 mm radialmente dal nervo

    # Lista per conservare i risultati di ogni singola fibra
    sfaps_registrati = []

    # 3. Iteriamo su ogni fibra della popolazione
    for diametro in diametri_fibre:
        # Inizializza il modello biofisico della fibra
        # (MRG è ottimo per le fibre afferenti mieliniche)
        fibra = MRGFiber(diameter=diametro, length=60000)  # Lunghezza totale 60mm

        # Configura lo stimolo extracellulare
        stimolo = PointSource(
            amplitude=-2.0,  # Ampiezza in mA
            pulse_width=0.1,  # Durata dell'impulso in ms
            position=(0, 0, z_stimolo)
        )
        fibra.add_stimulus(stimolo)

        # Configura l'elettrodo di registrazione alla distanza desiderata
        elettrodo = ExtracellularElectrode(
            position=(distanza_radiale, 0, z_registrazione)
        )

        # 4. Esecuzione della simulazione in NEURON tramite PyFibers
        sim = Simulator(fiber=fibra)
        risultato = sim.run()

        # Estrai il potenziale extracellulare (SFAP) nel punto di registrazione
        # basato sulle correnti di membrana generate dalla fibra
        sfap = elettrodo.calculate_potential(risultato.transmembrane_currents)
        sfaps_registrati.append(sfap)

    # 5. Calcolo del CAP sommando tutti gli SFAP
    # Converte la lista in un array NumPy e somma lungo l'asse delle fibre
    cap_totale = np.sum(np.array(sfaps_registrati), axis=0)

    return cap_totale


def genera_distribuzione_sensitiva(num_piccole, num_grandi):
    """
    Genera una distribuzione bimodale realistica per un nervo sensitivo.
    Restituisce un array NumPy con i diametri delle fibre in micrometri.
    """
    # 1. Parametri per le fibre piccole (es. tipo A-delta)
    media_piccole = 4.0  # Il picco è a 4 micrometri
    dev_std_piccole = 0.8  # Varianza attorno al picco

    # 2. Parametri per le fibre grandi (es. tipo A-beta)
    media_grandi = 10.0  # Il picco è a 10 micrometri
    dev_std_grandi = 1.2  # Varianza attorno al picco

    # 3. Generazione delle due popolazioni (distribuzioni Gaussiane)
    # np.random.normal(media, deviazione_standard, numero_di_elementi)
    fibre_piccole = np.random.normal(media_piccole, dev_std_piccole, num_piccole)
    fibre_grandi = np.random.normal(media_grandi, dev_std_grandi, num_grandi)

    # 4. Unione delle due popolazioni in un unico array
    tutte_le_fibre = np.concatenate([fibre_piccole, fibre_grandi])

    # 5. Pulizia dei dati (Filtro)
    # Rimuoviamo valori non realistici (es. sotto i 2 um o sopra i 20 um)
    # Questo previene anche errori nei modelli biofisici di PyFibers
    fibre_valide = tutte_le_fibre[(tutte_le_fibre >= 2.0) & (tutte_le_fibre <= 20.0)]

    return fibre_valide


# --- Istruzioni di implementazione ---

# Simuliamo un nervo con 300 fibre piccole e 200 fibre grandi
diametri_nervo = genera_distribuzione_sensitiva(num_piccole=300, num_grandi=200)

# Stampiamo quante fibre valide abbiamo ottenuto
print(f"Generati {len(diametri_nervo)} diametri di fibre validi.")

# Visualizziamo la distribuzione con un istogramma
plt.figure(figsize=(8, 5))
plt.hist(diametri_nervo, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Distribuzione Bimodale dei Diametri delle Fibre (Nervo Sensitivo)")
plt.xlabel("Diametro della fibra (um)")
plt.ylabel("Numero di fibre")
plt.grid(axis='y', alpha=0.75)
plt.show()

# Esecuzione della funzione (pronta per essere plottata con matplotlib)
# cap_segnale = simula_cap_afferente()