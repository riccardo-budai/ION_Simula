import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import sparse
from scipy.sparse.linalg import spsolve


# --- 1. FUNZIONI DI CONFIGURAZIONE MODELLO (Come prima) ---

def ottieni_coordinate_mne(montaggio='standard_1020'):
    """Ottiene coordinate reali C3, C4 e Cz."""
    mon = mne.channels.make_standard_montage(montaggio)
    posizioni = mon.get_positions()['ch_pos']
    return posizioni['C3'], posizioni['C1'], posizioni['Cz']


def crea_mesh_voxel_3d(dim=50):
    """Crea il volume 3D con conduttività diverse."""
    sigma = np.zeros((dim, dim, dim))
    c = dim // 2
    x, y, z = np.ogrid[:dim, :dim, :dim]
    raggio = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    r_testa = dim // 2 - 2

    # Assegnazione Conduttività (S/m)
    sigma[raggio < (r_testa - 7)] = 0.33    # Cervello
    mask_cranio = (raggio >= (r_testa - 7)) & (raggio < (r_testa - 2))
    sigma[mask_cranio] = 0.008              # Cranio (Isolante)
    mask_scalpo = (raggio >= (r_testa - 2)) & (raggio < r_testa)
    sigma[mask_scalpo] = 0.43               # Scalpo
    sigma[raggio >= r_testa] = 1e-10        # Aria
    #
    return sigma, r_testa


def mappa_coord_a_grid(coord_reale, r_testa_grid, dim_grid, centro_mne):
    r_mne = np.linalg.norm(centro_mne)
    fattore_scala = r_testa_grid / r_mne
    coord_grid = coord_reale * fattore_scala
    c = dim_grid // 2
    idx = np.array([coord_grid[0], coord_grid[1], coord_grid[2]]) + c
    return np.round(idx).astype(int)


# --- 2. SOLUTORE FEM ---

def costruisci_e_risolvi(sigma, anodo_idx, catodo_idx):
    """Costruisce la matrice e risolve per il Potenziale (Phi)."""
    nx, ny, nz = sigma.shape
    n = nx * ny * nz
    indices = np.arange(n)

    # Costruzione Matrice Sparsa (Metodo Vettoriale Semplificato)
    # Nota: Usiamo una conduttanza media semplice per velocità didattica
    vals = sigma.flatten()
    row_ind, col_ind, values = [], [], []
    diag_vals = np.zeros(n)

    shifts = [1, -1, nz, -nz, ny * nz, -(ny * nz)]  # Vicini: x, y, z

    print("Costruzione matrice sistema...")
    for shift in shifts:
        valid = np.ones(n, dtype=bool)
        if shift > 0:
            valid[-shift:] = False
        else:
            valid[:-shift] = False

        # Conduttanza tra i e vicino
        cond = (vals + np.roll(vals, shift)) / 2.0
        cond = cond[valid]

        rows = indices[valid]
        cols = indices[valid] + shift

        row_ind.append(rows)
        col_ind.append(cols)
        values.append(-cond)
        np.add.at(diag_vals, rows, cond)

    # Assemblaggio
    row_ind = np.concatenate(row_ind + [indices])
    col_ind = np.concatenate(col_ind + [indices])
    values = np.concatenate(values + [diag_vals])

    A = sparse.csr_matrix((values, (row_ind, col_ind)), shape=(n, n))

    # Termini noti
    b = np.zeros(n)
    idx_a = anodo_idx[0] * (ny * nz) + anodo_idx[1] * nz + anodo_idx[2]
    idx_c = catodo_idx[0] * (ny * nz) + catodo_idx[1] * nz + catodo_idx[2]
    b[idx_a] = 1.0
    b[idx_c] = -1.0

    print("Risoluzione equazione matriciale...")
    phi = spsolve(A, b).reshape((nx, ny, nz))
    #
    return phi


# --- 3. NUOVA LOGICA: CALCOLO CAMPO ELETTRICO ---

def calcola_campo_elettrico(phi, sigma):
    """
    Calcola E = -gradiente(Phi) e la densità J = sigma * E.
    Ritorna le componenti vettoriali.
    """
    # np.gradient ritorna d/daxis0, d/daxis1, d/daxis2
    # Nel nostro caso: axis0=X, axis1=Y, axis2=Z
    grad_x, grad_y, grad_z = np.gradient(phi)

    # E = - grad(Phi)
    Ex, Ey, Ez = -grad_x, -grad_y, -grad_z

    # J = sigma * E
    Jx, Jy, Jz = sigma * Ex, sigma * Ey, sigma * Ez

    return (Ex, Ey, Ez), (Jx, Jy, Jz)


def visualizza_vettori(sigma, phi, E_vec, slice_idx):
    """Visualizza Potenziale e Frecce del campo elettrico."""
    Ex, Ey, Ez = E_vec

    plt.figure(figsize=(12, 6))

    # Estraiamo le slice 2D al centro (vista coronale X-Z)
    # Assumiamo asse 1 (Y) come profondità
    sigma_slice = sigma[:, slice_idx, :].T
    phi_slice = phi[:, slice_idx, :].T

    # Vettori per la slice (X e Z)
    # Nota: Trasponiamo (.T) per allinearci con la visualizzazione imshow (righe=Z, col=X)
    Ex_slice = Ex[:, slice_idx, :].T
    Ez_slice = Ez[:, slice_idx, :].T

    # Creiamo griglia per le frecce
    Z, X = np.mgrid[0:sigma.shape[2], 0:sigma.shape[0]]

    # Plot 1: Potenziale
    plt.subplot(1, 2, 1)
    plt.imshow(phi_slice, origin='lower', cmap='jet', alpha=0.9)
    plt.title("Potenziale Elettrico (Phi)")
    plt.xlabel("X (Sinistra-Destra)")
    plt.ylabel("Z (Inferiore-Superiore)")
    plt.colorbar(label="V")

    # Plot 2: Campo Elettrico (Vettori)
    plt.subplot(1, 2, 2)
    # Mostriamo l'anatomia di sfondo in grigio
    plt.imshow(sigma_slice, origin='lower', cmap='gray_r', alpha=0.4)

    # Disegniamo le frecce (Quiver)
    # Sottocampioniamo (::2) per non avere troppe frecce
    step = 2
    plt.quiver(X[::step, ::step], Z[::step, ::step],
               Ex_slice[::step, ::step], Ez_slice[::step, ::step],
               color='red', scale=0.5, scale_units='xy')

    plt.title("Vettori Campo Elettrico (mathbf{E})")
    plt.xlabel("X (Sinistra-Destra)")

    # Zoomiamo sulla parte alta della testa per vedere meglio
    plt.ylim(10, sigma.shape[2])

    plt.tight_layout()
    plt.show()


def visualizza_con_target(sigma, J_mag, target_pos, slice_idx, score):
    plt.figure(figsize=(10, 5))

    # Sezione coronale
    img_sigma = sigma[:, slice_idx, :].T
    img_J = J_mag[:, slice_idx, :].T

    # Coordinate target proiettate sulla slice
    tx, tz = target_pos[0], target_pos[2]

    plt.imshow(img_J, origin='lower', cmap='inferno', vmax=np.percentile(img_J, 99))
    plt.colorbar(label="Densità di Corrente $|J|$")

    # Disegna il target (Cerchio verde)
    cerchio = plt.Circle((tx, tz), 4, color='lime', fill=False, linewidth=2, label='Target Corticale')
    plt.gca().add_patch(cerchio)

    plt.title(f"Stimolazione Corticale\nFocalità: {score:.2f}x (Target vs Background)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.tight_layout()
    plt.show()

def calcola_metrica_focalita(J_vec, sigma, pos_elettrodo, centro_testa, raggio_roi=4):
    """
    Calcola quanta corrente arriva nella regione corticale target rispetto al resto.
    """
    Jx, Jy, Jz = J_vec
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2 + Jz ** 2)

    # 1. Identifica il Target (Corteccia Motoria)
    # Il target non è l'elettrodo (che è sulla pelle), ma è SOTTO di esso.
    # Calcoliamo un vettore direzione: Elettrodo -> Centro Testa
    vettore_direzione = centro_testa - pos_elettrodo
    norma = np.linalg.norm(vettore_direzione)
    vettore_unitario = vettore_direzione / (norma + 1e-9)

    # Ci spostiamo verso l'interno di circa 6-8 voxel (attraversiamo scalpo e cranio)
    distanza_corteccia = 7.0
    pos_target = pos_elettrodo + (vettore_unitario * distanza_corteccia)
    pos_target = np.round(pos_target).astype(int)

    # 2. Crea maschera ROI (Sfera intorno al target)
    nx, ny, nz = J_mag.shape
    x, y, z = np.ogrid[:nx, :ny, :nz]
    dist_sq = (x - pos_target[0]) ** 2 + (y - pos_target[1]) ** 2 + (z - pos_target[2]) ** 2
    mask_roi = dist_sq <= raggio_roi ** 2

    # 3. Crea maschera Tutto il Cervello (dove sigma è ~0.33)
    # Escludiamo scalpo e cranio dal calcolo "rumore"
    mask_brain = (sigma > 0.30) & (sigma < 0.40)

    # 4. Calcoli
    # Intensità media nel bersaglio
    if np.sum(mask_roi) > 0:
        intensita_target = np.mean(J_mag[mask_roi])
    else:
        intensita_target = 0

    # Intensità media nel resto del cervello (background)
    # Escludiamo la ROI stessa dal background per enfatizzare il contrasto
    mask_bg = mask_brain & (~mask_roi)
    intensita_bg = np.mean(J_mag[mask_bg])

    # Score: Quante volte il target è più stimolato della media cerebrale?
    focalita = intensita_target / (intensita_bg + 1e-9)

    return focalita, pos_target, intensita_target


# --- 4. ESECUZIONE ---
if __name__ == "__main__":
    # Parametri
    N = 50  # Risoluzione

    # Setup MNE e Grid
    c3, c4, cz = ottieni_coordinate_mne()
    sigma, r_testa = crea_mesh_voxel_3d(N)
    c3_idx = mappa_coord_a_grid(c3, r_testa, N, cz)
    c4_idx = mappa_coord_a_grid(c4, r_testa, N, cz)

    # Risoluzione
    phi = costruisci_e_risolvi(sigma, c3_idx, c4_idx)

    # Calcolo Vettori
    # (Ex, Ey, Ez), (Jx, Jy, Jz) = calcola_campo_elettrico(phi, sigma)

    # Visualizzazione alla fetta Y dove si trova C3 (approssimativamente)
    # slice_coronale = c3_idx[1]

    # print(f"\nVisualizzazione slice Y={slice_coronale}. Nota la direzione delle frecce.")
    # visualizza_vettori(sigma, phi, (Ex, Ey, Ez), slice_coronale)

    # Ricalcoliamo J vettoriale
    (Ex, Ey, Ez), J_vec = calcola_campo_elettrico(phi, sigma)
    Jx, Jy, Jz = J_vec
    J_mag = np.sqrt(Jx ** 2 + Jy ** 2 + Jz ** 2)  # Magnitudine scalare

    # Centro della griglia
    centro_grid = np.array([N // 2, N // 2, N // 2])

    # CALCOLO FOCALITÀ SU C3 (Anodo)
    score, pos_target, int_target = calcola_metrica_focalita(J_vec, sigma, c3_idx, centro_grid)

    print(f"\n--- RISULTATI ANALISI ---")
    print(f"Posizione Elettrodo (Scalpo): {c3_idx}")
    print(f"Posizione Target Stimata (Corteccia): {pos_target}")
    print(f"Intensità Media sul Target: {int_target:.4f} A/m^2")
    print(f"PUNTEGGIO FOCALITÀ: {score:.2f}")

    if score < 1.5:
        print(">> GIUDIZIO: SCARSO. La corrente è troppo diffusa.")
    elif score > 2.5:
        print(">> GIUDIZIO: ECCELLENTE. Stimolazione ben localizzata.")
    else:
        print(">> GIUDIZIO: BUONO. Stimolazione efficace.")

    # Visualizza
    visualizza_con_target(sigma, J_mag, pos_target, c3_idx[1], score)
