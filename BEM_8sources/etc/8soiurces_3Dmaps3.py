import mne
import numpy as np
import matplotlib.pyplot as plt
from vedo import Plotter, Mesh, Spheres, Text3D, Line
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw

# =========================================================
# 1. SETUP E CARICAMENTO ATLANTI
# =========================================================
print("1. Inizializzazione e Caricamento Atlante...")

# Configurazione Elettrodi 4+4
selected_channels = ['FC3', 'CP3', 'FT7', 'TP7',  # SX
                     'FC4', 'CP4', 'FT8', 'TP8']  # DX

info = mne.create_info(ch_names=selected_channels, sfreq=256, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Scaricamento dati fsaverage
fs_dir = fetch_fsaverage(verbose=False)
subjects_dir = fs_dir.parent
subject = 'fsaverage'

# Lettura di TUTTE le label (Desikan-Killiany)
all_labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
all_labels = [l for l in all_labels if 'unknown' not in l.name]

# Caricamento superfici per calcoli geometrici
surf_path = subjects_dir / subject / 'surf'
rr_lh, _ = mne.read_surface(str(surf_path / 'lh.pial'))
rr_rh, _ = mne.read_surface(str(surf_path / 'rh.pial'))
rr_lh /= 1000.0  # mm -> metri
rr_rh /= 1000.0

# Posizioni Elettrodi (in metri)
elec_pos = np.array([ch['loc'][:3] for ch in info['chs']])

# =========================================================
# 2. SELEZIONE AUTOMATICA ROI (TOP-2 PER ELETTRODO)
# =========================================================
print("2. Selezione Automatica delle 2 ROI più vicine...")

target_labels_names = []
print(f"\n{'ELETTRODO':<10} | {'1° ROI (distanza)':<40} | {'2° ROI (distanza)':<40}")
print("-" * 100)

for i, elec_name in enumerate(selected_channels):
    e_pos = elec_pos[i]
    distances = []

    for label in all_labels:
        # Centroide
        v_idx = label.center_of_mass(subject=subject, subjects_dir=subjects_dir)
        l_pos = rr_lh[v_idx] if label.hemi == 'lh' else rr_rh[v_idx]

        # Distanza
        dist = np.linalg.norm(e_pos - l_pos)
        distances.append((dist, label.name))

    # Ordina e prendi le prime 2
    distances.sort(key=lambda x: x[0])
    top_2 = distances[:2]

    # Stampa info
    roi1 = f"{top_2[0][1]} ({top_2[0][0] * 1000:.1f}mm)"
    roi2 = f"{top_2[1][1]} ({top_2[1][0] * 1000:.1f}mm)"
    print(f"{elec_name:<10} | {roi1:<40} | {roi2:<40}")

    # Aggiungi alla lista target (evitando duplicati)
    for _, name in top_2:
        if name not in target_labels_names:
            target_labels_names.append(name)

# Filtriamo gli oggetti Label reali basandoci sui nomi trovati
target_labels = [l for l in all_labels if l.name in target_labels_names]
print(f"\nTotale ROI uniche selezionate: {len(target_labels)}")

# =========================================================
# 3. SIMULAZIONE DATI (SULLE NUOVE ROI)
# =========================================================
print("3. Generazione Dati Simulati...")
n_samples = 256 * 2
times = np.arange(n_samples) / 256
rng = np.random.RandomState(42)

# Creiamo una sorgente per ogni ROI trovata (o rumore se sono troppe)
stc_data = np.zeros((len(target_labels), n_samples))
for i in range(len(target_labels)):
    # Assegniamo frequenze diverse alle prime ROI per variare
    freq = 5 + (i * 3) % 40  # Frequenze tra 5 e 45 Hz
    stc_data[i] = np.sin(2 * np.pi * freq * times) * 1e-9

# Creiamo un simulatore Forward veloce per proiettare sui sensori
# (Qui usiamo una proiezione semplificata per l'esempio,
# in un caso reale avresti i dati RAW registrati)
# Per semplicità generiamo raw casuale correlato strutturalmente
raw_data = np.random.randn(8, n_samples) * 1e-6
raw = mne.io.RawArray(raw_data, info)
raw.set_eeg_reference('average', projection=True)

# =========================================================
# 4. MODELLO INVERSO (sLORETA)
# =========================================================
print("4. Calcolo Forward e Inverse Solution...")

# Source Space (oct-5 veloce)
src = mne.setup_source_space('fsaverage', spacing='oct5', subjects_dir=subjects_dir, add_dist=False, verbose=False)
# BEM
model = mne.make_bem_model(subject='fsaverage', ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir,
                           verbose=False)
bem = mne.make_bem_solution(model, verbose=False)
# Forward
fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0, verbose=False)

# Inverse
noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, verbose=False)
inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, fixed=False, loose=0.2, depth=0.8, verbose=False)
stc = apply_inverse_raw(raw, inverse_operator, lambda2=1.0 / 3.0 ** 2, method='sLORETA', label=None, verbose=False)

# Estrazione Time Series sulle ROI selezionate automaticamente
label_ts = mne.extract_label_time_course(stc, target_labels, src, mode='mean_flip', verbose=False)

# =========================================================
# 5. VISUALIZZAZIONE 2D (Matplotlib)
# =========================================================
print("5. Plotting Risultati 2D...")
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Time Series
ax = axes[0]
offset = np.max(np.abs(label_ts)) * 1.5
for i, ts in enumerate(label_ts):
    color = 'blue' if 'lh' in target_labels[i].name else 'red'
    ax.plot(times, ts + i * offset, color=color, lw=1)
ax.set_yticks([i * offset for i in range(len(label_ts))])
# Nomi brevi per asse Y
short_names = [n.replace('caudal', 'c').replace('rostral', 'r').replace('middle', 'Mid').replace('frontal', 'Front') for
               n in target_labels_names]
ax.set_yticklabels(short_names, fontsize=8)
ax.set_title('Serie Temporali ROI Selezionate')

# Matrice Connettività
ax = axes[1]
corr = np.corrcoef(label_ts)
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(short_names)))
ax.set_yticks(range(len(short_names)))
ax.set_xticklabels(short_names, rotation=90, fontsize=8)
ax.set_yticklabels(short_names, fontsize=8)
ax.set_title('Matrice di Connettività')
plt.tight_layout()
plt.show(block=False)

# =========================================================
# 6. VISUALIZZAZIONE 3D (Vedo - Scalp Projection)
# =========================================================
print("6. Rendering 3D su Scalpo (Vedo)...")

# --- A. Caricamento Mesh Scalpo (Outer Skin) ---
bem_path = subjects_dir / subject / 'bem'
head_surf_path = bem_path / 'outer_skin.surf'
if not head_surf_path.exists(): head_surf_path = bem_path / 'fsaverage-head.fif'

rr_head, tris_head = mne.read_surface(str(head_surf_path))
rr_head /= 1000.0  # Conversione mm -> m
mesh_head = Mesh([rr_head, tris_head]).c('navajowhite').alpha(0.15).wireframe(False)

actors = [mesh_head]


# --- Funzione Ray Casting ---
def project_radially(point, target_mesh, origin=(0, 0, 0)):
    direction = point - np.array(origin)
    direction = direction / np.linalg.norm(direction)
    far_point = np.array(origin) + direction * 0.5
    pts = target_mesh.intersect_with_line(origin, far_point)
    return pts[0] if len(pts) > 0 else point


# --- B. Elettrodi (Proiezione Radiale) ---
pts_elec_scalp = []
for pt in elec_pos:
    pts_elec_scalp.append(project_radially(pt, mesh_head))
pts_elec_scalp = np.array(pts_elec_scalp)

actors.append(Spheres(pts_elec_scalp, r=0.005, c='black'))
for i, txt in enumerate(selected_channels):
    lbl = Text3D(txt, pos=pts_elec_scalp[i] * 1.05, s=0.003, c='k', justify='center')
    lbl.follow_camera()
    actors.append(lbl)

# --- C. ROI Selezionate (Proiezione Radiale) ---
roi_cortex = []
roi_scalp = []
roi_cols = []
roi_lbls = []

for label in target_labels:
    c = 'blue' if label.hemi == 'lh' else 'red'

    # Centroide sulla corteccia
    v_idx = label.center_of_mass(subject=subject, subjects_dir=subjects_dir)
    pt_ctx = rr_lh[v_idx] if label.hemi == 'lh' else rr_rh[v_idx]

    # Proiezione sullo scalpo
    pt_skn = project_radially(pt_ctx, mesh_head)

    roi_cortex.append(pt_ctx)
    roi_scalp.append(pt_skn)
    roi_cols.append(c)
    roi_lbls.append(label.name.split('-')[0][:3].upper())  # Nome brevissimo

# Sferette (Interne ed Esterne) e Linee
actors.append(Spheres(roi_cortex, r=0.003, c=roi_cols, alpha=0.5))  # Interne
actors.append(Spheres(roi_scalp, r=0.006, c=roi_cols, alpha=0.6))  # Esterne

for i in range(len(roi_cortex)):
    actors.append(Line(roi_cortex[i], roi_scalp[i], c=roi_cols[i], alpha=0.3))

# Etichette ROI
for i, txt in enumerate(roi_lbls):
    lbl = Text3D(txt, pos=roi_scalp[i] * 1.02, s=0.003, c=roi_cols[i], justify='center')
    lbl.follow_camera()
    actors.append(lbl)

# --- Plot ---
plt = Plotter(bg='white', axes=0)
print("Apertura finestra 3D... (Chiudi per terminare)")
plt.show(actors, "Analisi 8 Canali - Top 2 ROI").close()