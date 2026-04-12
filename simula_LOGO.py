"""

"""

import mne
from vedo import Plotter, Mesh, settings
import platform

# ------------------------
# Impostazione per evitare problemi con alcuni backend grafici su Mac
if platform.system() == "Darwin":
    settings.default_backend = "vtk"

print("Sto cercando i dati 'fsaverage' di MNE...")

# 2. CARICAMENTO DATI (MNE)
# --------------------------
try:
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
except Exception as e:
    print(f"Errore durante il download di fsaverage: {e}")
    print("Assicurati di avere una connessione internet.")
    exit()

print(f"Dati trovati in: {fs_dir}")

lh_pial_path = fs_dir / 'surf' / 'lh.pial'
rh_pial_path = fs_dir / 'surf' / 'rh.pial'

# 3. CREAZIONE MESH (MNE + Vedo)
# ------------------------------
print("Caricamento superfici Pial...")

# Emisfero Sinistro
punti_lh, facce_lh = mne.read_surface(lh_pial_path)
lh_pial = Mesh([punti_lh, facce_lh])
lh_pial.c("grey")
lh_pial.name = "lh_pial"
# Aggiungiamo dati scalari per eventuali colorazioni future
# lh_pial.pointdata['MYPOINTARRAY'] = lh_pial.coordinates[:, 0]

# Emisfero Destro
punti_rh, facce_rh = mne.read_surface(rh_pial_path)
rh_pial = Mesh([punti_rh, facce_rh])
rh_pial.c("grey")
rh_pial.name = "rh_pial"
# rh_pial.pointdata['MYPOINTARRAY'] = rh_pial.coordinates[:, 0]

print("Mesh create con successo.")

# 4. CONFIGURAZIONE PLOTTER E ANIMAZIONE
# --------------------------------------
print("Avvio del visualizzatore interattivo...")

# Creiamo un unico Plotter
plt = Plotter(axes=0, title="R&B-Lab logo - Rotazione", bg='gray')


# --- Funzione di Rotazione ---
def rotate_logo(event):
    """
    Questa funzione viene chiamata ripetutamente dal timer.
    Ruota le mesh e aggiorna la scena.
    """
    # Ruota di 1 grado attorno all'asse Z (verticale)
    # axis=(0, 0, 1) indica l'asse Z. Puoi cambiare in (1,0,0) per X o (0,1,0) per Y.
    lh_pial.rotate(1.0, axis=(0, 0, 1))
    rh_pial.rotate(1.0, axis=(0, 0, 1))
    # Forza l'aggiornamento della grafica
    plt.render()


# Aggiungiamo la callback al plotter
plt.add_callback('timer', rotate_logo)

# Avviamo il timer
# dt=30 significa che la funzione viene chiamata circa ogni 30 millisecondi (circa 33 FPS)
plt.timer_callback('start', dt=30)

print("Finestra aperta. Premi 'q' o chiudi la finestra per uscire.")

# 5. VISUALIZZAZIONE
# ------------------
# interactive=True è fondamentale: mantiene la finestra aperta e attiva il loop degli eventi
plt.show(
    lh_pial,
    rh_pial,
    viewup='z',
    zoom=1.0,
    interactive=True
).close()

print("Visualizzatore chiuso.")