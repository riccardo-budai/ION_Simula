"""

"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import mne
from scipy import stats
from mne.minimum_norm import apply_inverse_raw
from mne.minimum_norm import read_inverse_operator
from mne.datasets import fetch_fsaverage
from vedo import Plotter, Mesh, Text3D, Spheres

path_eeg = '/media/ric23/Extreme SSD/Multimodal_dataset_hdf5'
file_path = os.path.join(path_eeg, 'sub-0564.h5')
print('path to eeg file =', file_path)

with h5py.File(file_path, 'r') as f:
    eeg = f['eeg_data'][:]  #, 0:10000]

print(eeg.shape, eeg.shape[1]/500, ' seconds')
print(f"avg amp = {np.mean(eeg, axis=0)}")

sampling_rate = 500
all_ch_names_in_file = ['FC3', 'FC4', 'CP3', 'CP4', 'FT7', 'FT8', 'TP7', 'TP8']
ch_map = {name: idx for idx, name in enumerate(all_ch_names_in_file)}
print("Mappa indici canali:", ch_map)

info = mne.create_info(all_ch_names_in_file, sampling_rate, 'eeg')
raw = mne.io.RawArray(eeg, info, verbose=False)

raw.filter(l_freq=0.5, h_freq=25, verbose=False)
raw.plot(block=True)


# Scarica fsaverage
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = fs_dir.parent
subject = 'fsaverage'

src = mne.setup_source_space('fsaverage', spacing='oct6', subjects_dir=subjects_dir, add_dist=False)

labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=subjects_dir)
# Nomi delle ROI target (corrispondenti agli 8 elettrodi)

target_labels_names = [
    # FC3
    'caudalmiddlefrontal-lh',   # (62.6mm)
    'superiorfrontal-lh',       # (74.5mm)
    # CP3
    'postcentral-lh',           # (57.9mm)
    'precentral-lh',            # (58.9mm)
    # FT7
    'parsopercularis-lh',       # (42.1mm)
    'rostralmiddlefrontal-lh',  # (47.7mm)
    # TP7
    'supramarginal-lh',         # (35.6mm)
    # 'postcentral-lh',           # (38.9mm)
    # FC4
    'caudalmiddlefrontal-rh',   # (62.8mm)
    'superiorfrontal-rh',       # (72.4mm)
    # CP4
    'precentral-rh',            # (56.2mm)
    'postcentral-rh',           # (57.3mm)
    # FT8
    'parsopercularis-rh',       # (41.1mm)
    'rostralmiddlefrontal-rh',      # (43.1mm)
    # TP8
    'supramarginal-rh'         # (28.5mm)
    # 'postcentral-rh'            # (36.8mm)
]
target_labels = [next(l for l in labels if l.name == name) for name in target_labels_names]

# Creazione montaggio
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)
raw.set_montage(montage)

# Generiamo dati con una struttura di segnale (sinusoidi) per vedere qualcosa di sensato
# invece di rumore bianco puro.

duration = 10
n_samples = sampling_rate * duration
times = np.arange(n_samples) / sampling_rate

beta_freq = 25
signal_beta = np.sin(2 * np.pi * beta_freq * times) * 1e-5  # Ampiezza ridotta (Beta è spesso bassa)

# B. ALFA (10 Hz) - Posteriore Sinistro
alpha_freq = 10
signal_alpha = np.sin(2 * np.pi * alpha_freq * times) * 2e-5  # Ampiezza media (Alfa è dominante)

# C. DELTA (2 Hz) - Posteriore Destro
delta_freq = 3
signal_delta = np.sin(2 * np.pi * delta_freq * times) * 5e-5  # Ampiezza alta (onde lente ampie)

# 4. Mixing sui Canali (Mappatura spaziale)
# -----------------------------------------------------------
data = np.zeros((8, n_samples))
rng = np.random.RandomState(42)

# 1. FRONTALE (Beta) - Simmetrico
# FC3 (Sx) e FC4 (Dx)
data[ch_map['FC3'], :] += signal_beta
data[ch_map['FC4'], :] += signal_beta

# Vicini Frontali/Temporali
data[ch_map['FT7'], :] += signal_beta # Sx
data[ch_map['FT8'], :] += signal_beta # Dx

# 2. POSTERIORE (Alpha) - Simmetrico
# Parietali
data[ch_map['CP3'], :] += signal_alpha # Sx
data[ch_map['CP4'], :] += signal_alpha # Dx

# Temporali Posteriori
data[ch_map['TP7'], :] += signal_alpha # Sx
data[ch_map['TP8'], :] += signal_alpha # Dx

# 5. Aggiunta Rumore e Creazione Raw
# Aggiungiamo rumore bianco di fondo per realismo
noise_level = 0.1 * np.mean(np.abs(signal_alpha))
data += rng.randn(8, n_samples) * noise_level

# raw = mne.io.RawArray(data, info)
raw = mne.io.RawArray(eeg, info)
raw.set_montage(montage)
raw.set_eeg_reference('average', projection=True)
raw.filter(l_freq=1.0, h_freq=15, verbose=False)


# --- CARICAMENTO (in un nuovo script) ---
inverse_operator = read_inverse_operator('modello_8ch-inv.fif')

# Calcolo sLORETA
snr = 3.0
lambda2 = 1.0 / snr ** 2
stc = apply_inverse_raw(raw, inverse_operator, lambda2, method='sLORETA')   #'sLORETA' dSPM)

'''
fwd = mne.read_forward_solution('modello_8ch-fwd.fif')
print("Creazione Covarianza Artificiale Simmetrica...")
# Crea una matrice diagonale: dice che tutti i sensori hanno lo stesso rumore standard
noise_cov = mne.make_ad_hoc_cov(raw.info)

# Ti serve il forward model. Se non l'hai salvato, ricalcolalo o leggilo se esiste
# Se non hai il file fwd, copia le righe di make_forward_solution dal file model
# fwd = mne.read_forward_solution('modello_8ch-fwd.fif')

# Calcola l'operatore inverso "ideale"
inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info,
    fwd,
    noise_cov,
    loose=0.2,
    depth=0.8
)
# Calcolo sLORETA
snr = 3.0
lambda2 = 1.0 / snr ** 2
# Ora applica questo operatore
stc = apply_inverse_raw(raw, inverse_operator, lambda2, method='sLORETA')
'''

# Estrazione Time Series (Unmixing)
# mode='mean_flip' è cruciale: inverte i segni dei dipoli per evitare che
# segnali positivi e negativi nella stessa ROI si cancellino a vicenda.
label_ts = mne.extract_label_time_course(stc, target_labels, src, mode='mean_flip')

# label_ts ha forma (n_labels, n_times) -> (8, 512)
'''
# =========================================================
# 4. VISUALIZZAZIONE FINALE (Nomi ROI su Asse Y e Matrice)
# =========================================================
print("4. Visualizzazione...")

fig, axes = plt.subplots(2, 1, figsize=(15, 15))  # Altezza adeguata per 2 grafici

# ---------------------------------------------------------
# Grafico 1: Serie Temporali con Nomi ROI sull'Asse Y
# ---------------------------------------------------------
ax = axes[0]
offset_step = np.max(np.abs(label_ts)) * 1.5  # Spaziatura verticale
colors = {'lh': 'blue', 'rh': 'red'}  # Blu=Sx, Rosso=Dx

# Creiamo nomi brevi per la visualizzazione (più puliti)
short_names = [name.replace('caudalmiddlefrontal', 'CMFront')
               .replace('supramarginal', 'SupMarg')
               .replace('parsopercularis', 'ParsOp')
               .replace('parstringularis', 'ParsTri')
               .replace('precentral', 'PreCtr')
               .replace('superiorfrontal', 'SupFront')
               .replace('postcentral', 'PostCtr')
               .replace('-lh', ' (L)').replace('-rh', ' (R)')
               for name in target_labels_names]

# Plot dei segnali
for i, ts in enumerate(label_ts):
    # Determina colore
    hemi = 'lh' if '-lh' in target_labels_names[i] else 'rh'

    # Plotta il segnale traslato di 'i * offset_step'
    ax.plot(times, ts + i * offset_step,
            color=colors[hemi], linewidth=1.5, alpha=0.9)

    # Opzionale: aggiungi una linea tratteggiata per lo zero di ogni segnale
    ax.axhline(i * offset_step, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

# --- QUI AGGIUNGIAMO I NOMI SULL'ASSE Y ---
# Impostiamo i tick dell'asse Y esattamente alle posizioni degli offset
ax.set_yticks([i * offset_step for i in range(len(label_ts))])
# Assegniamo i nomi delle ROI a questi tick
ax.set_yticklabels(short_names, fontsize=10)        #, fontweight='bold')

# Titoli e Legenda semplificata
ax.set_title('Attività ROI: Sinistra (Blu) vs Destra (Rosso)', fontsize=14)
ax.set_xlabel('Tempo (s)')
ax.set_ylabel('Regioni di Interesse (ROI)')  # L'asse Y ora rappresenta le regioni

# Legenda solo per i colori (non per le singole ROI, che sono già sull'asse Y)
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', lw=2)]
ax.legend(custom_lines, ['Emisfero Sinistro', 'Emisfero Destro'], bbox_to_anchor=(1.02, 1), loc='upper left')

ax.grid(True, axis='x', linestyle='--', alpha=0.6)  # Griglia solo verticale per tempo
ax.margins(x=0.01)  # Riduci margini laterali

# ---------------------------------------------------------
# Grafico 2: Matrice di Connettività (Invariato)
# ---------------------------------------------------------
ax = axes[1]
corr_matrix = np.corrcoef(label_ts)

im = ax.imshow(corr_matrix, interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_title('Matrice di Connettività (Correlazione di Pearson)', fontsize=14)

# Etichette assi
ax.set_xticks(np.arange(len(short_names)))
ax.set_yticks(np.arange(len(short_names)))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(short_names, fontsize=10)

# Valori numerici nelle celle
for i in range(len(short_names)):
    for j in range(len(short_names)):
        text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
        ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                ha="center", va="center", color=text_color, fontsize=9) # , weight='bold')

plt.colorbar(im, ax=ax, label='Correlazione', fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
'''

# =========================================================
# 5. VISUALIZZAZIONE 2D (DUE FINESTRE SEPARATE)
# =========================================================
print("5. Generazione Grafici 2D in finestre separate...")

# --- FINESTRA 1: Serie Temporali ---
plt.figure(figsize=(10, 8), num='Serie Temporali ROI') # 'num' dà il titolo alla finestra
ax1 = plt.gca() # Ottieni l'asse corrente della nuova figura

offset_step = np.max(np.abs(label_ts)) * 1.5 if np.max(np.abs(label_ts)) > 0 else 1e-9
short_names_ts = [n.replace('caudal', 'c').replace('rostral', 'r').replace('middle', 'Mid').replace('frontal', 'Front').replace('temporal', 'Temp') for n in target_labels_names]

times = np.arange(eeg.shape[1]) / sampling_rate
for i, ts in enumerate(label_ts):
    color = 'blue' if 'lh' in target_labels_names[i] else 'red'
    ax1.plot(times, ts + i * offset_step, color=color, lw=1.5, alpha=0.9)

ax1.set_yticks([i * offset_step for i in range(len(label_ts))])
ax1.set_yticklabels(short_names_ts, fontsize=9, fontweight='bold')
ax1.set_title('Attività Temporale ROI (Blu=Sx, Rosso=Dx)', fontsize=14)
ax1.set_xlabel('Tempo (s)')
ax1.grid(True, axis='x', linestyle='--', alpha=0.5)

# Legenda colori
# custom_lines = [Line2D([0], [0], color='blue', lw=2), Line2D([0], [0], color='red', lw=2)]
# ax1.legend(custom_lines, ['Sinistra', 'Destra'], loc='upper right')

plt.tight_layout()
plt.show(block=False) # Mostra la prima finestra e continua l'esecuzione

# --- FINESTRA 2: Matrice Connettività (Significativa p < 0.05) ---
plt.figure(figsize=(9, 8), num='Matrice Connettività Significativa')
ax2 = plt.gca()

n_rois = len(label_ts)
corr_matrix = np.zeros((n_rois, n_rois))
p_values = np.zeros((n_rois, n_rois))

# Calcolo R e P-value per ogni coppia
for i in range(n_rois):
    for j in range(n_rois):
        if i == j:
            corr_matrix[i, j] = 1.0
            p_values[i, j] = 0.0
        else:
            r, p = stats.pearsonr(label_ts[i], label_ts[j])
            corr_matrix[i, j] = r
            p_values[i, j] = p

# Creiamo una maschera per nascondere i valori NON significativi
# (p >= 0.05) oppure correlazioni molto deboli
significance_level = 0.05
mask_non_sig = p_values >= significance_level

# Creiamo una versione "mascherata" della matrice per il plot
# I valori non significativi diventeranno bianchi/trasparenti
corr_masked = np.ma.masked_where(mask_non_sig, corr_matrix)

# Plot
# 'bad' color = white (per i valori mascherati)
cmap = plt.cm.RdBu_r
cmap.set_bad('white')

im = ax2.imshow(corr_masked, cmap=cmap, vmin=-1, vmax=1)
plt.colorbar(im, ax=ax2, label=f'Correlazione (p < {significance_level})', fraction=0.046, pad=0.04)

# Assi e Etichette
ax2.set_xticks(range(len(short_names_ts)))
ax2.set_yticks(range(len(short_names_ts)))
ax2.set_xticklabels(short_names_ts, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(short_names_ts, fontsize=9)
ax2.set_title(f'Connettività Significativa (p < {significance_level})', fontsize=14)

# Valori nelle celle (Solo se significativi)
for i in range(len(short_names_ts)):
    for j in range(len(short_names_ts)):
        # Mostra testo solo se p < 0.05
        if p_values[i, j] < significance_level:
            txt_col = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
            # Aggiungi un asterisco per p molto bassi
            asterisk = "**" if p_values[i, j] < 0.001 else "*" if p_values[i, j] < 0.01 else ""
            label_text = f"{corr_matrix[i, j]:.2f}\n{asterisk}"

            ax2.text(j, i, label_text, ha="center", va="center", color=txt_col, fontsize=7)
        else:
            # Opzionale: scrivi "ns" (not significant) o lascia vuoto
            ax2.text(j, i, "ns", ha="center", va="center", color="gray", fontsize=6)
            pass

# Griglia per separare le celle (utile se ci sono molti buchi bianchi)
ax2.set_xticks(np.arange(n_rois + 1) - .5, minor=True)
ax2.set_yticks(np.arange(n_rois + 1) - .5, minor=True)
ax2.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
ax2.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.show(block=True)

# =========================================================
# 5. VISUALIZZAZIONE 3D (Elettrodi e ROI)
# =========================================================
'''
from mpl_toolkits.mplot3d import Axes3D

print("5. Generazione grafico 3D...")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# --- A. PLOT ELETTRODI (Triangoli Neri) ---
# Estraiamo le coordinate (x, y, z) dal montage caricato
# raw.info['chs'] contiene i dizionari per ogni canale con la chiave 'loc'
elec_pos = np.array([ch['loc'][:3] for ch in raw.info['chs']])

# Scatter plot degli elettrodi
ax.scatter(elec_pos[:, 0], elec_pos[:, 1], elec_pos[:, 2],
           c='k', s=100, marker='^', label='Elettrodi (4+4)', depthshade=False)

# Etichette elettrodi
for i, name in enumerate(raw.info['ch_names']):
    ax.text(elec_pos[i, 0], elec_pos[i, 1], elec_pos[i, 2] + 0.005,
            name, fontsize=10, fontweight='bold', color='black')

# --- B. PLOT ROI (Pallini Colorati) ---
# Calcoliamo il centroide geometrico di ogni ROI usando i vertici del Source Space
# (Nota: src contiene le coordinate 'rr' dei vertici della materia grigia)

for i, label in enumerate(target_labels):
    # Determina emisfero (0 = Sinistra, 1 = Destra) per accedere a src corretto
    hemi_idx = 0 if label.hemi == 'lh' else 1
    color = 'blue' if label.hemi == 'lh' else 'red'

    # Trova quali vertici della Label esistono nel Source Space ridotto (oct6)
    # (Non tutti i vertici dell'atlante sono nel source space sparso)
    common_vertices = np.intersect1d(label.vertices, src[hemi_idx]['vertno'])

    if len(common_vertices) > 0:
        # Prendi le coordinate 3D di questi vertici
        # src[hemi]['rr'] sono tutte le coordinate, usiamo common_vertices come indici?
        # No, common_vertices sono indici globali della mesh originale.
        # Dobbiamo mapparli. src[hemi]['rr'] ha forma (N_vertices_totali, 3).
        # Possiamo accedere direttamente con gli indici dei vertici.
        roi_points = src[hemi_idx]['rr'][common_vertices]

        # Calcola il centroide (media x, y, z)
        centroid = np.mean(roi_points, axis=0)

        # Plot del centroide
        ax.scatter(centroid[0], centroid[1], centroid[2],
                   c=color, s=200, marker='o', alpha=0.6, edgecolors='w')

        # Nome ROI semplificato
        short_name = target_labels_names[i].split('-')[0][:4].upper()
        ax.text(centroid[0], centroid[1], centroid[2],
                short_name, fontsize=8, color=color, ha='center')

# --- C. ESTETICA E LEGENDA ---
ax.set_title('Localizzazione Spaziale: Elettrodi vs ROI', fontsize=14)
ax.set_xlabel('X (Sinistra-Destra)')
ax.set_ylabel('Y (Posteriore-Anteriore)')
ax.set_zlabel('Z (Inferiore-Superiore)')

# Rimuovi i piani grigi per pulizia (opzionale)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Legenda personalizzata
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='^', color='w', label='Elettrodi', markerfacecolor='k', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='ROI Sinistra', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='ROI Destra', markerfacecolor='red', markersize=10)
]
ax.legend(handles=legend_elements, loc='upper right')

# Imposta una vista iniziale comoda (es. vista dall'alto-sinistra)
ax.view_init(elev=30, azim=135)
'''
# plt.tight_layout()
# plt.show()

########################################################################################################################

# Assumiamo che tu abbia già nel tuo workspace:
# - target_labels: la lista delle 15-16 ROI (oggetti Label MNE)
# - label_ts: le serie temporali estratte (n_roi x n_time)
# - subjects_dir, subject: i percorsi fsaverage

# Se non li hai caricati, decommenta queste righe per un test rapido:
# (assicurati di aver eseguito le parti precedenti per avere target_labels e label_ts)
# ...

print("7. Mappatura Colore su Corteccia (Heatmap Connettività)...")

# ---------------------------------------------------------
# A. Calcolo dell'Indice da Visualizzare (Node Strength)
# ---------------------------------------------------------
# Calcoliamo la matrice di correlazione
corr_matrix = np.corrcoef(label_ts)
np.fill_diagonal(corr_matrix, 0)  # Ignoriamo l'autocorrelazione (che è sempre 1)

# Calcoliamo la "Node Strength": media delle correlazioni assolute per ogni ROI
# Questo ci dice: "Quanto è importante questo nodo nella rete?"

# roi_values = np.mean(np.abs(corr_matrix), axis=1)
roi_values = np.mean(np.abs(label_ts), axis=1)

# Normalizziamo tra 0 e 1 per la colormap (opzionale, vedo lo fa in automatico ma aiuta)
# roi_values = (roi_values - roi_values.min()) / (roi_values.max() - roi_values.min())

print("Valori calcolati per ROI (Node Strength):")
for i, val in enumerate(roi_values):
    print(f"  {target_labels[i].name}: {val:.3f}")

# ---------------------------------------------------------
# B. Preparazione Mesh e Array Scalari
# ---------------------------------------------------------
surf_path = subjects_dir / subject / 'surf'

# Leggiamo le superfici originali (High Resolution) per avere tutti i vertici
# Nota: Le label di MNE sono definite sulla superficie originale di fsaverage (~160k vertici)
rr_lh, tris_lh = mne.read_surface(str(surf_path / 'lh.pial'))
rr_rh, tris_rh = mne.read_surface(str(surf_path / 'rh.pial'))

# Conversione mm -> metri
rr_lh /= 1000.0
rr_rh /= 1000.0

# Inizializziamo gli array dei valori (scalari) a 0 (o NaN)
# Saranno usati per colorare: 0 = Grigio (sfondo), >0 = Colore ROI
scalars_lh = np.zeros(len(rr_lh))
scalars_rh = np.zeros(len(rr_rh))

# ---------------------------------------------------------
# C. "Dipingiamo" i Vertici delle ROI
# ---------------------------------------------------------
for i, label in enumerate(target_labels):
    val = roi_values[i]

    # I vertici della label sono indici che puntano alla mesh originale
    vertices = label.vertices

    if label.hemi == 'lh':
        # Assegniamo il valore a tutti i vertici che compongono questa ROI
        # Usiamo un "trucco": aggiungiamo un valore minimo (es. 0.01) se il valore è 0
        # per distinguerlo dallo sfondo in caso di colormap che parte da 0.
        scalars_lh[vertices] = val
    else:
        scalars_rh[vertices] = val

# ---------------------------------------------------------
# D. Visualizzazione con Vedo
# ---------------------------------------------------------
actors = []

# Creazione Mesh LH
mesh_lh = Mesh([rr_lh, tris_lh])
# Assegniamo i dati scalari ai punti (vertici)
mesh_lh.pointdata['Connectivity'] = scalars_lh

# Applichiamo la Colormap
# 'jet', 'viridis', 'hot', 'coolwarm'.
# vmin/vmax controllano il range. Impostiamo vmin leggermente sopra 0 per lasciare il cervello grigio
mesh_lh.cmap('jet', 'Connectivity', vmin=np.min(roi_values) * 0.9, vmax=np.max(roi_values))
# Rendiamo grigio tutto ciò che è sotto la soglia (lo sfondo)
mesh_lh.add_scalarbar(title='Mean Correlation')

# Creazione Mesh RH
mesh_rh = Mesh([rr_rh, tris_rh])
mesh_rh.pointdata['Connectivity'] = scalars_rh
mesh_rh.cmap('jet', 'Connectivity', vmin=np.min(roi_values) * 0.9, vmax=np.max(roi_values))

# Uniamo tutto
actors.append(mesh_lh)
actors.append(mesh_rh)

# --- Opzionale: Aggiungiamo etichette 3D sui picchi ---
for i, label in enumerate(target_labels):
    # Centroide per il testo
    v_idx = label.center_of_mass(subject=subject, subjects_dir=subjects_dir)
    pt = rr_lh[v_idx] if label.hemi == 'lh' else rr_rh[v_idx]

    # Testo con il valore numerico
    txt = f"{label.name.split('-')[0][:3]}\n{roi_values[i]:.2f}"
    actors.append(Text3D(txt, pos=pt * 1.01, s=0.002, c='k', justify='center'))

# plt = Plotter(bg='white', axes=0)
# print("Apertura Mappa Corticale Colorata...")
# plt.show(actors, "Mappa di Connettività ROI su Superficie Piale").close()

# =========================================================
# 4. AGGIUNTA TESTA (SCALPO) E ELETTRODI
# =========================================================
print("4. Aggiunta Scalpo ed Elettrodi...")

# A. Caricamento Scalpo (Outer Skin)
bem_path = subjects_dir / subject / 'bem'
head_surf_path = bem_path / 'outer_skin.surf'
if not head_surf_path.exists(): head_surf_path = bem_path / 'fsaverage-head.fif'

rr_head, tris_head = mne.read_surface(str(head_surf_path))
rr_head /= 1000.0 # mm -> metri

mesh_head = Mesh([rr_head, tris_head]).c('navajowhite').alpha(0.15).wireframe(False)
actors.append(mesh_head)

# B. Posizionamento Elettrodi
elec_pos = np.array([ch['loc'][:3] for ch in info['chs']]) # Già in metri

# Proiezione robusta elettrodi sulla pelle (Closest Point iterativo)
pts_elec_scalp = []
for pt in elec_pos:
    # Trova il punto più vicino sulla mesh della testa
    proj = mesh_head.closest_point(pt)
    pts_elec_scalp.append(proj)
pts_elec_scalp = np.array(pts_elec_scalp)

# Sfere Elettrodi (Nere)
# actors.append(Spheres(pts_elec_scalp, r=0.005, c='gray'))

# Etichette Elettrodi
for i, txt in enumerate(all_ch_names_in_file):
    # Testo fluttuante sopra l'elettrodo
    lbl = Text3D(txt, pos=pts_elec_scalp[i] + np.array([0,0,0.008]), s=0.003, c='k', justify='center')
    lbl.follow_camera()
    # actors.append(lbl)

# =========================================================
# 5. VISUALIZZAZIONE FINALE
# =========================================================
print("5. Apertura Finestra 3D...")
plt = Plotter(bg='white', axes=1)

# Aggiungiamo una barra scala per la colormap (solo su una mesh per pulizia)
mesh_lh.add_scalarbar(title='Avg Amplitude', nlabels=3)

plt.show(actors, "Mappa Connettività Corticale + Elettrodi").close()