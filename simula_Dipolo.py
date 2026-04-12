import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- PARAMETRI FISICI ---
SIGMA_E = 0.3 # Conducibilità extracellulare del tessuto (S/m)
DIPOLE_LENGTH_M = 100e-6 # Lunghezza del dipolo (100 micron)
CURRENT_MICROA = 50e-6   # Corrente totale del pool (50 microAmpere)

# --- DEFINIZIONE DELLO SPAZIO ---
# Coordinate fisse del dipolo (Pool neuronale centrato in 0,0,0)
DIPOLE_POSITIONS = {
    'Source': np.array([0.0, 0.0, DIPOLE_LENGTH_M / 2]),   # +Z
    'Sink':   np.array([0.0, 0.0, -DIPOLE_LENGTH_M / 2])   # -Z
}

# --- FUNZIONE DI CALCOLO ---

def calculate_field_potential(electrode_pos, current, sigma_e, dipole_pos_dict):
    """
    Calcola il potenziale di campo (FP) in una data posizione dell'elettrodo
    risultante da un dipolo di corrente (Source e Sink).

    Args:
        electrode_pos (np.array): Posizione 3D dell'elettrodo (x, y, z).
        current (float): Ampiezza della corrente (I).
        sigma_e (float): Conducibilità extracellulare.
        dipole_pos_dict (dict): Dizionario delle posizioni Source e Sink.

    Returns:
        float: Potenziale di campo (V).
    """
    K = 1.0 / (4 * np.pi * sigma_e)
    FP_total = 0.0

    # Potenziale dalla Sorgente (+I)
    r_source = np.linalg.norm(electrode_pos - dipole_pos_dict['Source'])
    if r_source > 0: # Evita la divisione per zero
        FP_total += K * (current / r_source)

    # Potenziale dal Pozzo (-I)
    r_sink = np.linalg.norm(electrode_pos - dipole_pos_dict['Sink'])
    if r_sink > 0:
        FP_total += K * (-current / r_sink)

    return FP_total

# --- ESECUZIONE DELLA SIMULAZIONE ---

# Vettore di posizioni per l'elettrodo lungo l'asse Z (profondità)
Z_range = np.linspace(-500e-6, 500e-6, 50) # Da -500 a +500 micron
X_fixed = 100e-6 # Distanza orizzontale fissa dall'asse Z (100 micron)
Y_fixed = 0.0

FP_values = []
for z in Z_range:
    electrode_position = np.array([X_fixed, Y_fixed, z])
    fp = calculate_field_potential(electrode_position, CURRENT_MICROA, SIGMA_E, DIPOLE_POSITIONS)
    FP_values.append(fp)

FP_values = np.array(FP_values) * 1e3 # Converti da Volt a milliVolt (mV)

# --- INTERPRETAZIONE ---
print(f"La massima escursione del potenziale (FP) è: {np.max(FP_values) - np.min(FP_values):.2f} mV")

# --- VISUALIZZAZIONE DEI RISULTATI ---
plt.figure(figsize=(8, 5))
plt.plot(Z_range * 1e6, FP_values, marker='o', linestyle='-', markersize=4)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(DIPOLE_POSITIONS['Source'][2] * 1e6, color='r', linestyle=':', label='Sorgente (+)')
plt.axvline(DIPOLE_POSITIONS['Sink'][2] * 1e6, color='b', linestyle=':', label='Pozzo (-)')

plt.title('Potenziale di Campo (fPSP) lungo l\'asse Z (Distanza orizzontale = 100μm)')
plt.xlabel('Profondità (micrometri $\mu m$)')
plt.ylabel('Potenziale di Campo (mV)')
plt.grid(True)
plt.legend()
plt.show()

