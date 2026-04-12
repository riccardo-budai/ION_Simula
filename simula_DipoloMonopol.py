
import numpy as np
import matplotlib.pyplot as plt

# --- PARAMETRI FISICI ---
SIGMA_E = 0.3 # Conducibilità extracellulare del tessuto (S/m)
DIPOLE_LENGTH_M = 100e-6 # Lunghezza del dipolo (100 micron)
CURRENT_MICROA = 500e-6  # Corrente aumentata per un segnale visibile a 4cm (500 microAmpere)

# --- DISTANZA ELETTRDO FISSA (4 cm) ---
DISTANCE_X_M = 0.01 # 4 centimetri, convertiti in metri (0.04 m)

# --- DEFINIZIONE DELLO SPAZIO (DIPOLO SULL'ASSE X) ---

# Il centro del dipolo è a (0, 0, 0)
DIPOLE_POSITIONS = {
    # NUOVA POSIZIONE: Allineamento sull'asse X (Antero-Posteriore)
    'Source': np.array([DIPOLE_LENGTH_M / 2, 0.0, 0.0]),   # +X
    'Sink':   np.array([-DIPOLE_LENGTH_M / 2, 0.0, 0.0])   # -X
}

# --- FUNZIONE DI CALCOLO (Invariata) ---

def calculate_field_potential(electrode_pos, current, sigma_e, dipole_pos_dict):
    """
    Calcola il potenziale di campo (FP) in una data posizione dell'elettrodo.
    """
    K = 1.0 / (4 * np.pi * sigma_e)
    FP_total = 0.0

    # Contributo dalla Sorgente (+I)
    r_source = np.linalg.norm(electrode_pos - dipole_pos_dict['Source'])
    if r_source > 0:
        FP_total += K * (current / r_source)

    # Contributo dal Pozzo (-I)
    r_sink = np.linalg.norm(electrode_pos - dipole_pos_dict['Sink'])
    if r_sink > 0:
        FP_total += K * (-current / r_sink)

    return FP_total

# --- SIMULAZIONE DELLA REGISTRAZIONE MONOPOLARE ---

# L'elettrodo è fisso in X e Y (a 4 cm di distanza) e si muove lungo Z (profondità)
Z_range = np.linspace(-0.01, 0.01, 200) # Range di profondità: da -1 cm a +1 cm

# L'elettrodo è posizionato a 4 cm di distanza orizzontale
X_fixed = DISTANCE_X_M
Y_fixed = 0.0

FP_monopolar_values = []
for z in Z_range:
    # La posizione dell'elettrodo è [4cm, 0, Z]
    electrode_position = np.array([X_fixed, Y_fixed, z])
    fp = calculate_field_potential(electrode_position, CURRENT_MICROA, SIGMA_E, DIPOLE_POSITIONS)
    FP_monopolar_values.append(fp)

FP_values_uV = np.array(FP_monopolar_values) * 1e6 # Converti da Volt a microVolt (uV)

# --- VISUALIZZAZIONE DEI RISULTATI ---
plt.figure(figsize=(8, 5))
plt.plot(Z_range * 1e2, FP_values_uV,
         linestyle='-', label=f'Registrazione a X={DISTANCE_X_M * 100:.1f} cm')

plt.axhline(0, color='gray', linestyle='--')

plt.title(f'Potenziale di Campo (FP) con Dipolo sull\'Asse X (AP)')
plt.xlabel('Profondità Z (cm)')
plt.ylabel(r'Potenziale di Campo ($\mu V$)')
plt.grid(True)
plt.legend()
plt.show()

'''
# --- PARAMETRI FISICI ---
SIGMA_E = 0.3 # Conducibilità extracellulare del tessuto (S/m)
DIPOLE_LENGTH_M = 100e-6 # Lunghezza del dipolo (100 micron)
CURRENT_MICROA = 50e-6   # Corrente totale del pool (50 microAmpere)

# --- DEFINIZIONE DELLO SPAZIO ---
# Coordinate fisse del dipolo (Pool neuronale centrato in 0,0,0)
DIPOLE_POSITIONS = {
    'Source': np.array([0.0, 0.0, DIPOLE_LENGTH_M / 2]),   # Fonte di corrente (+Z)
    'Sink':   np.array([0.0, 0.0, -DIPOLE_LENGTH_M / 2])   # Pozzo di corrente (-Z)
}

# --- FUNZIONE DI CALCOLO (Invariata) ---

def calculate_field_potential(electrode_pos, current, sigma_e, dipole_pos_dict):
    """
    Calcola il potenziale di campo (FP) in una data posizione dell'elettrodo
    risultante da un dipolo di corrente (Source e Sink).
    """
    K = 1.0 / (4 * np.pi * sigma_e)
    FP_total = 0.0

    # Contributo dalla Sorgente (+I)
    r_source = np.linalg.norm(electrode_pos - dipole_pos_dict['Source'])
    if r_source > 0:
        FP_total += K * (current / r_source)

    # Contributo dal Pozzo (-I)
    r_sink = np.linalg.norm(electrode_pos - dipole_pos_dict['Sink'])
    if r_sink > 0:
        # FP_total += K * (-current / r_sink)
        pass

    return FP_total

# --- SIMULAZIONE DELLA REGISTRAZIONE MONOPOLARE IDEALE ----------------------------------------------------------------

# Vettore di posizioni per l'elettrodo lungo l'asse Z (profondità)
Z_range = np.linspace(-500e-6, 500e-6, 100) # Da -500 a +500 micron
X_fixed = 0.0 # Posizioniamo l'elettrodo direttamente sopra l'asse del dipolo
Y_fixed = 0.0

FP_monopolar_values = []
for z in Z_range:
    electrode_position = np.array([X_fixed, Y_fixed, z])
    fp = calculate_field_potential(electrode_position, CURRENT_MICROA, SIGMA_E, DIPOLE_POSITIONS)
    FP_monopolar_values.append(fp)

FP_monopolar_values = np.array(FP_monopolar_values) * 1e3 # Converti da Volt a milliVolt (mV)

# --- VISUALIZZAZIONE DEI RISULTATI ---
plt.figure(figsize=(8, 5))
plt.plot(Z_range * 1e6, FP_monopolar_values,
         marker='o', linestyle='-', markersize=3, label='Registrazione Monopolare (Attivo vs Rif. Lontano)')

plt.axhline(0, color='gray', linestyle='--')

# Linee di riferimento
plt.axvline(DIPOLE_POSITIONS['Source'][2] * 1e6, color='r', linestyle=':', label='Sorgente (+)')
plt.axvline(DIPOLE_POSITIONS['Sink'][2] * 1e6, color='b', linestyle=':', label='Pozzo (-)')

plt.title('Potenziale di Campo Monopolare Ideale Lungo l\'asse Z')
plt.xlabel('Profondità (micrometri $\mu m$)')
plt.ylabel('Potenziale di Campo (mV)')
plt.grid(True)
plt.legend()
plt.show()
'''