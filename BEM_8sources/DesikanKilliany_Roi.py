import mne
import numpy as np
from vedo import Plotter, Mesh, Spheres, Text3D
from mne.datasets import fetch_fsaverage
import random

# =========================================================
# 1. SETUP: Caricamento Dati Standard MNE
# =========================================================
print("1. Caricamento Modello Anatomico (fsaverage)...")
fs_dir = fetch_fsaverage(verbose=False)
subjects_dir = fs_dir.parent
subject = 'fsaverage'

actors = []
# Percorsi ai file standard
surf_path = subjects_dir / subject / 'surf'
bem_path = subjects_dir / subject / 'bem'

# A. Caricamento TESTA (Scalpo)
head_path = bem_path / 'outer_skin.surf'
if not head_path.exists(): head_path = bem_path / 'fsaverage-head.fif'

rr_head, tris_head = mne.read_surface(str(head_path))
rr_head /= 1000.0  # mm -> Metri

rr_lh, tris_lh = mne.read_surface(str(surf_path / 'lh.pial'))
rr_rh, tris_rh = mne.read_surface(str(surf_path / 'rh.pial'))
rr_lh /= 1000.0  # mm -> Metri
rr_rh /= 1000.0

# Mesh Testa Trasparente (alpha=0.15)
mesh_head = Mesh([rr_head, tris_head]).c('navajowhite').alpha(0.15).wireframe(False)
actors.append(mesh_head)

mesh_lh = Mesh([rr_lh, tris_lh]).c('gray').alpha(0.15).wireframe(False)
actors.append(mesh_lh)
mesh_rh = Mesh([rr_rh, tris_rh]).c('gray').alpha(0.15).wireframe(False)
actors.append(mesh_rh)


# =========================================================
# 2. CONFIGURAZIONE ELETTRODI
# =========================================================
print("2. Configurazione Elettrodi...")
selected_channels = ['FC3', 'CP3', 'FT7', 'TP7', 'FC4', 'CP4', 'FT8', 'TP8']
info = mne.create_info(ch_names=selected_channels, sfreq=256, ch_types='eeg')
info.set_montage(mne.channels.make_standard_montage('standard_1020'))
elec_pos = np.array([ch['loc'][:3] for ch in info['chs']])

# =========================================================
# 3. CREAZIONE SCENA VEDO
# =========================================================


# Mesh Testa (Trasparente)
mesh_head = Mesh([rr_head, tris_head]).c('navajowhite').alpha(0.15).wireframe(False)
actors.append(mesh_head)

print("3. Generazione Mesh ROI Allineate con Colori Simmetrici...")

# Lista ROI desiderate
target_rois = [
    'caudalmiddlefrontal-lh', 'superiorfrontal-lh', 'postcentral-lh', 'supramarginal-lh',
    'parsopercularis-lh', 'precentral-lh', 'rostralmiddlefrontal-lh',
    'caudalmiddlefrontal-rh', 'superiorfrontal-rh', 'postcentral-rh', 'supramarginal-rh',
    'parsopercularis-rh', 'precentral-rh', 'rostralmiddlefrontal-rh'
]

# --- LOGICA COLORI SIMMETRICI ---
roi_color_map = {}
distinct_colors = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF",
    "#FF8000", "#800080", "#008080", "#FF0080", "#80FF00", "#0080FF",
    "#A52A2A", "#D2691E", "#DC143C", "#20B2AA", "#778899", "#B0C4DE"
]
col_idx = 0

labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)

for label in labels:
    if label.name in target_rois:
        # 1. Determina il nome "base" (senza -lh o -rh)
        base_name = label.name.replace('-lh', '').replace('-rh', '')

        # 2. Assegna o recupera il colore
        if base_name in roi_color_map:
            # Colore già assegnato all'altro emisfero
            assigned_color = roi_color_map[base_name]
        else:
            # Nuovo colore
            if col_idx < len(distinct_colors):
                assigned_color = distinct_colors[col_idx]
                col_idx += 1
            else:
                assigned_color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            roi_color_map[base_name] = assigned_color

        # 3. Scegli i vertici corretti
        if label.hemi == 'lh':
            rr = rr_lh
        else:
            rr = rr_rh

        # 4. Crea la Mesh ROI
        roi_pts = rr[label.vertices]
        # Creiamo un "point cloud" denso e poi ricostruiamo la superficie
        # Nota: usiamo il colore assegnato sia per i punti che per la superficie finale
        roi_mesh = Mesh([roi_pts, []]).c(assigned_color)

        try:
            # Ricostruzione superficie (può essere pesante, radius piccolo = più dettagli)
            roi_surf = roi_mesh.reconstruct_surface(dims=(50, 50, 50), radius=0.01).alpha(1.0)
            roi_surf.color(assigned_color)
            roi_surf.caption(base_name)  # Aggiunge nome ROI se passi col mouse (opzionale)
            actors.append(roi_surf)
        except Exception as e:
            print(f"Warn: Skip {label.name} ricostruzione fallita ({e})")

# =========================================================
# 4. PROIEZIONE ELETTRODI SU SCALPO
# =========================================================
print("4. Proiezione Elettrodi...")
pts_projected = []
for pt in elec_pos:
    proj_pt = mesh_head.closest_point(pt)
    pts_projected.append(proj_pt)

pts_projected = np.array(pts_projected)
actors.append(Spheres(pts_projected, r=0.005, c='red'))

for i, name in enumerate(selected_channels):
    lbl = Text3D(name, pos=pts_projected[i] * 1.05, s=0.003, c='k', justify='center')
    lbl.follow_camera()
    # actors.append(lbl)

# =========================================================
# 5. VISUALIZZAZIONE
# =========================================================
print("5. Avvio Viewer...")
plt = Plotter(bg='white', axes=1)
plt.show(actors, "Analisi ROI Simmetriche").close()
