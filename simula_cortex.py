

import mne

from vedo import Plotter, Mesh, settings, dataurl, Cube, VedoLogo
import platform

# Impostazione per evitare problemi con alcuni backend grafici
if platform.system() == "Darwin":
    settings.default_backend = "vtk"

print("Sto cercando i dati 'fsaverage' di MNE...")

# 1. CARICAMENTO DATI (MNE)
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
lh_inflated_path = fs_dir / 'surf' / 'lh.inflated'
rh_inflated_path = fs_dir / 'surf' / 'rh.inflated'

# 2. CARICAMENTO MESH (MNE + Vedo)
# ----------------------------------------------------
print("Caricamento superfici Pial...")
# Emisfero Sinistro Pial
punti_lh_pial, facce_lh_pial = mne.read_surface(lh_pial_path)
lh_pial = Mesh([punti_lh_pial, facce_lh_pial])
lh_pial.color = "grey"
lh_pial.name = "lh_pial"
lh_pial.pointdata['MYPOINTARRAY'] = lh_pial.coordinates[:,0]
print(f"punti mesh LH = {lh_pial}")

# Emisfero Destro Pial
punti_rh_pial, facce_rh_pial = mne.read_surface(rh_pial_path)
rh_pial = Mesh([punti_rh_pial, facce_rh_pial])
rh_pial.color = "grey"
rh_pial.name = "rh_pial"
rh_pial.pointdata['MYPOINTARRAY'] = rh_pial.coordinates[:,0]
print(f"punti mesh RH = {rh_pial}")


print("Caricamento superfici Inflated...")
# Emisfero Sinistro Inflated
punti_lh_infl, facce_lh_infl = mne.read_surface(lh_inflated_path)
lh_inflated = Mesh([punti_lh_infl, facce_lh_infl])
lh_inflated.color = "lightgrey"
lh_inflated.name = "lh_inflated"
lh_inflated.off()


# Emisfero Destro Inflated
punti_rh_infl, facce_rh_infl = mne.read_surface(rh_inflated_path)
rh_inflated = Mesh([punti_rh_infl, facce_rh_infl])
rh_inflated.color = "lightgrey"
rh_inflated.name = "rh_inflated"
rh_inflated.off()

print("Caricamento completato.")


# 3. DEFINIZIONE DELLE FUNZIONI DI CALLBACK
# -----------------------------------------
# <-- SEZIONE MODIFICATA -->
# Accediamo a plt.sliders[0][0] invece di plt.sliders[0]

def toggle_hemisphere(widget, ename):
    widget.switch()

    current_text = widget.status()
    is_visible = "ON" in current_text.upper()

    hemi_name = widget.name.lower()
    current_surf_type = plt.buttons[0].status().lower()
    #
    for mesh_s in plt.get_meshes():
        if mesh_s.name.startswith(hemi_name + "_") and mesh_s.name.endswith(current_surf_type):
            # mesh.visibility(is_visible)
            if is_visible:
                mesh_s.on()
            else:
                mesh_s.off()
    # vedo.printc("STATUS="+bt_surface.status(), box="_", dim=True)

def switch_surface(widget, event):
    lh_text = plt.buttons[1].status()
    rh_text = plt.buttons[2].status()

    is_lh_visible = "ON" in lh_text.upper()
    is_rh_visible = "ON" in rh_text.upper()

    visible_count = int(is_lh_visible) + int(is_rh_visible)

    if visible_count != 1:
        print("-------------------------")
        print(f"Azione 'Switch Superficie' bloccata: {visible_count} emisferi visibili.")
        print("Per favore, mostra solo UN emisfero per cambiare superficie.")
        return  # Interrompe la funzione qui

    bt_surface.switch()
    current_status = widget.status()

    # --- ISTRUZIONI DI DEBUG ---
    print("-------------------------")
    print(f"Bottone superficie premuto! Nuovo stato: {current_status}")
    print(f"button LH: {plt.buttons[1].states} (Visibile? {is_lh_visible})")
    print(f"button RH: {plt.buttons[2].states} (Visibile? {is_rh_visible})")
    # ---------------------------

    if current_status == 'Pial':
        lh_inflated.off()
        rh_inflated.off()
        if is_lh_visible: lh_pial.on()
        if is_rh_visible: rh_pial.on()

    elif current_status == 'Inflated':
        lh_pial.off()
        rh_pial.off()
        if is_lh_visible: lh_inflated.on()
        if is_rh_visible: rh_inflated.on()


# 4. CREAZIONE DELLA GUI E DEL PLOTTER
# ------------------------------------
# <-- SEZIONE MODIFICATA -->
# Torniamo a passare una LISTA di 2 punti a 'pos'

print("Avvio del visualizzatore interattivo...")

plt = Plotter(axes=1, title="Visualizzatore Cervello MNE + Vedo", bg='gray')
              # bg=dataurl+"images/tropical.jpg")
              # bg=dataurl+"res/logos/logo1.png")

# A. Bottone per cambiare superficie (invariato)
bt_surface = plt.add_button(
    switch_surface,
    states=["Inflated", "Pial"],
    c=["w", "w"],
    bc=["blue", "green"],
    pos=(0.7, 0.05),
    size=12,
    font="Arial",
    bold=True,
)

# B. Bottoni per emisferi
# Bottone Sinistro (diventa plt.buttons[1])
bt_lh = plt.add_button(
    toggle_hemisphere,
    states=["LH: OFF", "LH: ON"], # Stati per il toggle
    c=["w", "w"],
    bc=["darkred", "grey"],     # Colori per ON e OFF
    pos=(0.8, 0.05),            # Posizione (in basso a sinistra)
    size=12,
    font="Arial",
    bold=True
)
bt_lh.name = "LH"

# Bottone Destro (diventa plt.buttons[2])
bt_rh = plt.add_button(
    toggle_hemisphere,
    states=["RH: OFF", "RH: ON"], # Stati per il toggle
    c=["w", "w"],
    bc=["darkblue", "grey"],    # Colori per ON e OFF
    pos=(0.9, 0.05),            # Posizione (accanto a LH)
    size=12,
    font="Arial",
    bold=True
)
bt_rh.name = "RH"

# 5. AVVIO
# --------
cid0 = plt.at(0).add_hover_legend()

plt.show(
    lh_pial, rh_pial,
    lh_inflated, rh_inflated,
    interactive=True,
    zoom=1.0
).close()

print("Visualizzatore chiuso.")