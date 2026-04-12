import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyfibers import build_fiber, FiberModel, ScaledStim


def genera_diametri_test(num_piccole, num_grandi):
    """Genera una piccola distribuzione bimodale per testare il codice."""
    piccole = np.random.normal(5.0, 0.5, num_piccole)
    grandi = np.random.normal(12.0, 1.0, num_grandi)
    fibre_valide = np.concatenate([piccole, grandi])
    return fibre_valide[(fibre_valide >= 2.0) & (fibre_valide <= 20.0)]


# ==========================================
# 1. PARAMETRI DI GEOMETRIA E STIMOLAZIONE
# ==========================================
# 1101 sezioni (100 internodi) garantiscono che la fibra
# sia abbastanza lunga per coprire i 40 mm di registrazione.
sezioni_fibra = 1101

z_stimolo = 10000.0  # Stimoliamo a 10 mm (10000 µm) dall'inizio della fibra
z_registrazione = 40000.0  # Registriamo a 40 mm (40000 µm). Distanza di propagazione = 30 mm!

# Impostazioni stimolo (dal tuo file)
time_step = 0.001
time_stop = 15  # 15 ms dovrebbero bastare per far viaggiare il segnale per 30 mm
start, on, off = 0, 0.1, 0.2
waveform = interp1d([start, on, off, time_stop], [0, 1, 0, 0], kind="previous")
stimamp = -9.5  # mA

# ==========================================
# 2. INIZIALIZZAZIONE DELLA SIMULAZIONE
# ==========================================
# Generiamo 10 fibre di test (5 piccole, 5 grandi)
diametri_nervo = genera_diametri_test(5, 5)

cap_totale = None
asse_tempi = None

print(f"Inizio simulazione per {len(diametri_nervo)} fibre...")

# ==========================================
# 3. CICLO DI GENERAZIONE DEL CAP
# ==========================================
for i, diametro in enumerate(diametri_nervo):
    print(f"Simulazione fibra {i + 1}/{len(diametri_nervo)} (Diametro: {diametro:.2f} µm)")

    # 1. PRIMA costuiamo la fibra! (Questo risolve l'errore di NEURON)
    fibra = build_fiber(FiberModel.MRG_INTERPOLATION, diameter=diametro, n_sections=sezioni_fibra)

    # 2. DOPO inizializziamo la stimolazione
    # (Usiamo le variabili time_step, time_stop e waveform che hai definito all'inizio)
    stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

    # Applica i potenziali extracellulari
    fibra.potentials = fibra.point_source_potentials(0, 250, z_stimolo, 1, 10)

    # Registriamo le correnti per calcolare l'SFAP dopo
    fibra.record_im(allsec=True)
    fibra.record_vext()

    # Esegui la simulazione
    ap, tempo_ap = stimulation.run_sim(stimamp, fibra)

    if ap > 0:
        # Calcoliamo i "potenziali reciproci" all'elettrodo di registrazione
        potenziali_reciproci = fibra.point_source_potentials(0, 1000, z_registrazione, 1, 1)

        # Estraiamo lo SFAP
        sfap_trace, time_trace = fibra.record_sfap(rec_potentials=potenziali_reciproci)

        # Inizializza l'array del CAP al primo ciclo valido
        if cap_totale is None:
            cap_totale = np.zeros_like(sfap_trace)
            asse_tempi = time_trace

        # Somma questo SFAP al CAP totale
        cap_totale += sfap_trace
    else:
        print(f"  -> Nessun potenziale d'azione generato per la fibra {i + 1}.")

# ==========================================
# 4. VISUALIZZAZIONE DEL RISULTATO
# ==========================================
if cap_totale is not None:
    plt.figure(figsize=(10, 5))
    plt.plot(asse_tempi, cap_totale, color='darkblue', linewidth=2)
    plt.title(f"Compound Action Potential (CAP)\nRegistrato a {(z_registrazione - z_stimolo) / 1000} mm dallo stimolo")
    plt.xlabel("Tempo (ms)")
    plt.ylabel("Ampiezza (µV)")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Aggiungiamo un marker per indicare l'istante dello stimolo
    plt.axvline(x=start, color='red', linestyle=':', label='Stimolo')
    plt.legend()
    plt.show()
else:
    print("Errore: Nessuna fibra ha generato un potenziale d'azione. Prova ad aumentare 'stimamp'.")
