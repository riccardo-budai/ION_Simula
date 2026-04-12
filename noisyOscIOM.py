import numpy as np
import matplotlib.pyplot as plt


def noisy_harmonic_oscillator(T, dt, x0, v0, omega_0, gamma, sigma):
    """
    Simula un oscillatore armonico rumoroso usando il metodo di Euler-Maruyama.

    Args:
        T (float): Tempo totale di simulazione.
        dt (float): Passo temporale (time step).
        x0 (float): Posizione iniziale.
        v0 (float): Velocità iniziale.
        omega_0 (float): Frequenza angolare naturale (omega).
        gamma (float): Coefficiente di smorzamento.
        sigma (float): Intensità del rumore.

    Returns:
        tuple: (array dei tempi, array delle posizioni)
    """
    N = int(T / dt)  # Numero totale di passi
    t = np.linspace(0, T, N)
    x = np.zeros(N)
    v = np.zeros(N)

    x[0] = x0
    v[0] = v0

    # Termine stocastico (Wiener process increment)
    dW = np.random.normal(0.0, np.sqrt(dt), N)

    for i in range(N - 1):
        # 1. Calcolo della variazione deterministica (Drift)
        dx_det = v[i] * dt
        dv_det = (-gamma * v[i] - omega_0 ** 2 * x[i]) * dt

        # 2. Calcolo della variazione stocastica (Diffusion)
        # La forma più semplice per la SDE di secondo ordine è applicare
        # il rumore solo alla variabile di accelerazione (dv).
        dv_stoch = sigma * dW[i]

        # 3. Aggiornamento delle variabili (Euler-Maruyama)
        x[i + 1] = x[i] + dx_det
        v[i + 1] = v[i] + dv_det + dv_stoch

    return t, x


# --- PARAMETRI DI SIMULAZIONE -----------------------------------------------------------------------------------------
T_total = 100.0  # Tempo totale (s)
dt_step = 0.01  # Passo temporale (s)

# Parametri fisici
frequency = 1.0  # Frequenza naturale (Hz)
omega_0 = 2 * np.pi * frequency
gamma_damp = 0.5  # Smorzamento (damping)

# Parametri iniziali
x_start = 1.0  # Posizione iniziale
v_start = 0.0  # Velocità iniziale

# --- ESECUZIONE DELLA SIMULAZIONE ---

# Caso 1: Basso Rumore (Oscillatore Quasi-Ideale)
sigma_low = 0.15
t_low, x_low = noisy_harmonic_oscillator(T_total, dt_step, x_start, v_start, omega_0, gamma_damp, sigma_low)

# Caso 2: Alto Rumore (Oscillatore Caotico)
sigma_high = 2.0
t_high, x_high = noisy_harmonic_oscillator(T_total, dt_step, x_start, v_start, omega_0, gamma_damp, sigma_high)

# --- VISUALIZZAZIONE ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(t_low, x_low, label=f'sigma = {sigma_low}', color='blue')
plt.title('Oscillatore Armonico con Basso Rumore')
plt.xlabel('Tempo (s)')
plt.ylabel('Posizione ($x$)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_high, x_high, label=f'sigma = {sigma_high}', color='orange')
plt.title('Oscillatore Caotico (armonico con alto rumore)')
plt.xlabel('Tempo (s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()