import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import fetch_fsaverage

# =========================================================
# 1. SETUP DATI (Elettrodi e ROI)
# =========================================================
# Elettrodi 4+4
selected_channels = ['FC3', 'CP3', 'FT7', 'TP7',  # Sinistra
                     'FC4', 'CP4', 'FT8', 'TP8']  # Destra

# ROI Selezionate (corrispondenti agli elettrodi)
target_labels_names = [
    'caudalmiddlefrontal-lh', 'supramarginal-lh', 'parsopercularis-lh', 'middletemporal-lh',  # Sx
    'caudalmiddlefrontal-rh', 'supramarginal-rh', 'parsopercularis-rh', 'middletemporal-rh'  # Dx
]

# Nomi brevi per il grafico
short_names_roi = ['CMF', 'SM', 'ParsOp', 'MidTemp', 'CMF', 'SM', 'ParsOp', 'MidTemp']

# Creazione Info e Montaggio
info = mne.create_info(ch_names=selected_channels, sfreq=256, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# =========================================================
# 2. RECUPERO COORDINATE 3D
# =========================================================
print("Recupero coordinate anatomiche...")

# A) Coordinate Elettrodi
# Estraiamo le posizioni 3D dal montaggio (in metri)
# Le chiavi sono i nomi dei canali
ch_pos = montage.get_positions()['ch_pos']
elec_coords = np.array([ch_pos[ch] for ch in selected_channels])

# B) Coordinate ROI (Centro di Massa)
# Scarichiamo fsaverage e le etichette
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = fs_dir.parent
labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=subjects_dir)

# Prepariamo lo spazio sorgente per mappare i vertici alle coordinate
src = mne.setup_source_space('fsaverage', spacing='oct6', subjects_dir=subjects_dir, add_dist=False)

roi_coords = []
for name in target_labels_names:
    # Trova la label
    label = next(l for l in labels if l.name == name)

    # Calcola il centro di massa (restituisce l'indice del vertice)
    # restrict_vertices=True assicura che il vertice sia dentro la label
    com_vertex_idx = label.center_of_mass(subject='fsaverage', subjects_dir=subjects_dir, restrict_vertices=True)

    # Trova la coordinata 3D reale del vertice nello spazio sorgente
    # src[0] è LH, src[1] è RH
    hemi_idx = 0 if label.hemi == 'lh' else 1
    # coordinate del vertice (in metri)
    vertex_coord = src[hemi_idx]['rr'][com_vertex_idx]
    roi_coords.append(vertex_coord)

roi_coords = np.array(roi_coords)

# =========================================================
# 3. VISUALIZZAZIONE 2D (Proiezione Assiale)
# =========================================================
# Usiamo una proiezione semplice X-Y (Vista dall'alto: Naso in alto, Orecchie ai lati)
# MNE Coordinates: X = Destra, Y = Anteriore (Naso), Z = Alto

fig, ax = plt.subplots(figsize=(10, 10))

# --- Disegno della Testa (Cerchio Schematico) ---
# Stimiamo il raggio della testa basandoci sugli elettrodi
head_radius = 0.095  # ~9.5 cm in metri
head_circle = plt.Circle((0, 0), head_radius, color='k', fill=False, linewidth=2)
ax.add_artist(head_circle)

# Naso (Triangolo in alto)
ax.plot([0, -0.01, 0.01, 0], [head_radius, head_radius + 0.01, head_radius + 0.01, head_radius], 'k')

# --- Plot Elettrodi (Cerchi) ---
# Dividiamo per colore Sx/Dx
for i, (x, y, z) in enumerate(elec_coords):
    color = 'blue' if '3' in selected_channels[i] or '7' in selected_channels[i] else 'red'
    ax.scatter(x, y, s=150, c=color, edgecolors='k', zorder=10, marker='o', label='Elettrodi' if i == 0 else "")
    # Etichetta Elettrodo (spostata leggermente)
    ax.text(x, y + 0.005, selected_channels[i], fontsize=11, ha='center', fontweight='bold', color=color)

# --- Plot ROI (Quadrati) ---
for i, (x, y, z) in enumerate(roi_coords):
    color = 'cyan' if 'lh' in target_labels_names[i] else 'magenta'
    # Plot del punto
    ax.scatter(x, y, s=150, c=color, edgecolors='k', zorder=5, marker='s', alpha=0.7,
               label='ROI Center' if i == 0 else "")
    # Etichetta ROI
    ax.text(x, y - 0.008, short_names_roi[i], fontsize=9, ha='center', color='darkslategray', style='italic')

    # --- Linea di Connessione (Elettrodo -> ROI Associata) ---
    # Disegniamo una linea tratteggiata per mostrare la relazione spaziale
    ax.plot([elec_coords[i, 0], roi_coords[i, 0]],
            [elec_coords[i, 1], roi_coords[i, 1]],
            color='gray', linestyle=':', alpha=0.5)

# --- Finiture ---
ax.set_aspect('equal')
ax.set_title('Mappa 2D: Elettrodi (Cerchi) e Centri ROI (Quadrati)', fontsize=15)
ax.set_xlim(-0.12, 0.12)
ax.set_ylim(-0.12, 0.12)
ax.axis('off')  # Nascondi assi cartesiani

# Legenda personalizzata
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Elettrodi Sx', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Elettrodi Dx', markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan', label='ROI Sx', markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='magenta', label='ROI Dx', markersize=10),
    Line2D([0], [0], color='gray', linestyle=':', label='Associazione')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.show()

