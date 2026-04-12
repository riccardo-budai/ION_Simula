import mne
import numpy as np
from vedo import Plotter, Mesh, Spheres, Text3D
from mne.datasets import fetch_fsaverage

# =========================================================
# 1. SETUP DATI E CONFIGURAZIONE
# =========================================================
print("1. Configurazione Elettrodi e ROI...")

# A. Elettrodi 4+4
selected_channels = ['FC3', 'CP3', 'FT7', 'TP7',  # SX
                     'FC4', 'CP4', 'FT8', 'TP8']  # DX
info = mne.create_info(ch_names=selected_channels, sfreq=256, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# B. Download fsaverage files (se non presenti)
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = fs_dir.parent
subject = 'fsaverage'

# C. Definizione ROI Target
target_labels_names = [
    'caudalmiddlefrontal-lh', 'caudalmiddlefrontal-rh',
    'supramarginal-lh', 'supramarginal-rh',
    'parsopercularis-lh', 'parsopercularis-rh',
    'middletemporal-lh', 'middletemporal-rh'
]
# Carica le labels
labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
target_labels = [l for l in labels if l.name in target_labels_names]

# =========================================================
# 2. CARICAMENTO MESH VEDO (Metodo Robusto via MNE)
# =========================================================
print("2. Caricamento Mesh Piali...")

surf_path = str(subjects_dir / subject / 'surf')

# 1. Usiamo MNE per leggere i file .pial (è infallibile per questo formato)
# rr = coordinate vertici, tris = indici triangoli
surf, tris_surf = mne.read_surface(surf_path + '/lh.seghead')
rr_lh, tris_lh = mne.read_surface(surf_path + '/lh.pial')
rr_rh, tris_rh = mne.read_surface(surf_path + '/rh.pial')

surf = surf /1000.0
rr_lh = rr_lh / 1000.0
rr_rh = rr_rh / 1000.0

# 2. Creiamo le Mesh Vedo partendo dai dati numerici
# Mesh([vertices, faces])
mesh_surf = Mesh([surf, tris_surf]).c('white').alpha(0.3)
mesh_lh = Mesh([rr_lh, tris_lh]).c('white').alpha(0.3)
mesh_rh = Mesh([rr_rh, tris_rh]).c('white').alpha(0.3)

actors = [mesh_surf, mesh_lh, mesh_rh]

# =========================================================
# 3. CREAZIONE OGGETTI 3D
# =========================================================
print("3. Calcolo Posizioni e Oggetti...")

# --- A. ELETTRODI (Proiezione Manuale Iterativa) ---
# Estraiamo posizioni originali
elec_pos = np.array([ch['loc'][:3] for ch in info['chs']])

pts_projected = []
for pt in elec_pos:
    # Trova punto più vicino su ogni emisfero separatamente
    p_lh = mesh_lh.closest_point(pt)
    p_rh = mesh_rh.closest_point(pt)
    p_surf = mesh_surf.closest_point(pt)

    # Calcola distanze
    d_lh = np.linalg.norm(pt - p_lh)
    d_rh = np.linalg.norm(pt - p_rh)
    d_surf = np.linalg.norm(pt - p_surf)

    # Scegli il migliore
    '''
    if d_lh < d_rh:
        pts_projected.append(p_lh)
    else:
        pts_projected.append(p_rh)
    '''

    pts_projected.append(p_surf)

pts_projected = np.array(pts_projected)

# Sfere per elettrodi
elec_spheres = Spheres(pts_projected, r=0.005, c='green')
actors.append(elec_spheres)

# Etichette Elettrodi (Loop correttivo)
for i, txt in enumerate(selected_channels):
    lbl = Text3D(txt, pos=pts_projected[i], s=0.003, c='k', justify='center')
    lbl.follow_camera()
    actors.append(lbl)

# --- B. ROI (Centroidi Colorati) ---
roi_cens = []
roi_colors = []
roi_names = []

for label in target_labels:
    c = 'blue' if label.hemi == 'lh' else 'red'

    # Calcolo vertice centrale della label (MNE method)
    # Restituisce l'indice del vertice sulla superficie originale
    v_idx = label.center_of_mass(subject=subject, subjects_dir=subjects_dir)

    # FIX: Usiamo direttamente gli array numpy caricati da MNE (rr_lh/rr_rh)
    # Invece di chiedere a vedo 'mesh_lh.points()[v_idx]' che può variare tra versioni
    if label.hemi == 'lh':
        pt = rr_lh[v_idx]
    else:
        pt = rr_rh[v_idx]

    roi_cens.append(pt)
    roi_colors.append(c)

    # Nome breve per visualizzazione
    name = label.name.split('-')[0][:4].upper()
    roi_names.append(name)

# Sfere colorate per ROI
roi_spheres = Spheres(roi_cens, r=0.008, c=roi_colors, alpha=0.8)
actors.append(roi_spheres)

# Etichette ROI (Loop correttivo anche qui per sicurezza)
for i, txt in enumerate(roi_names):
    # Offset leggero verso l'alto (z) per non sovrapporsi alla sfera
    pos_txt = roi_cens[i] + np.array([0, 0, 0.006])
    lbl = Text3D(txt, pos=pos_txt, s=0.004, c=roi_colors[i], justify='center')
    lbl.follow_camera()
    # actors.append(lbl)

# =========================================================
# 4. VISUALIZZAZIONE FINALE
# =========================================================
print("4. Apertura finestra Vedo...")

plt = Plotter(bg='white', axes=0)
plt.show(actors, "Localizzazione 8 Elettrodi vs ROI").close()