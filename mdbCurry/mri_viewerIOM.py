import os
import numpy as np
import vedo
from vedo.applications import RayCastPlotter, Slicer3DPlotter


class CurryMRIViewer:
    def __init__(self, metadata):
        """
        metadata: Dizionario contenente i dati letti da _attributes
                  (xsize, ysize, zsize, xscale, folder_path, ecc.)
        """
        self.meta = metadata
        self.volume = None

        # Estrai dimensioni (Voxel)
        try:
            self.nx = int(self.meta.get('xsize', 256))
            self.ny = int(self.meta.get('ysize', 256))
            self.nz = int(self.meta.get('zsize', 256))  # Numero di slice teorico

            # Estrai Spacing (mm per voxel)
            self.sx = float(self.meta.get('xscale', 1.0))
            self.sy = float(self.meta.get('yscale', 1.0))
            self.sz = float(self.meta.get('zscale', 1.0))

            # Percorso dei file
            self.path = self.meta.get('_full_path', '')

            # Header da saltare (in bytes)
            self.header_size = int(self.meta.get('header_size', 0))

            # Bit depth (16 bit solitamente per MRI)
            self.bits = int(self.meta.get('bits_store', 16))

        except ValueError as e:
            print(f"Errore nel parsing dei metadati: {e}")

    def load_data(self):
        """
        Legge i file slice dalla cartella e crea un array 3D Numpy.
        """
        if not os.path.exists(self.path):
            print("Cartella non trovata.")
            return False

        # 1. Trova i file slice
        # Assumiamo che siano tutti i file NON _attributes e non nascosti
        files = [f for f in os.listdir(self.path) if not f.startswith('.') and f != '_attributes']

        # 2. Ordinamento naturale (fondamentale!)
        # Altrimenti i.10 viene prima di i.2
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', s)]

        files.sort(key=natural_sort_key)

        # Controllo coerenza
        if len(files) == 0:
            print("Nessun file trovato.")
            return False

        print(f"Caricamento {len(files)} slices da: {self.path}")

        # 3. Allocazione Memoria (Array 3D vuoto)
        # MRI di solito è int16 o uint16
        dtype = np.int16 if self.bits == 16 else np.uint8

        # Creiamo il contenitore: [NumeroSlice, Altezza, Larghezza]
        # Nota: L'ordine Z, Y, X dipende spesso dal software. Curry spesso usa Z, Y, X.
        volume_data = np.zeros((len(files), self.ny, self.nx), dtype=dtype)

        # 4. Lettura Binaria Slice per Slice
        for i, filename in enumerate(files):
            filepath = os.path.join(self.path, filename)

            with open(filepath, 'rb') as f:
                # Salta l'header se presente
                if self.header_size > 0:
                    f.seek(self.header_size)

                # Legge i dati grezzi
                raw_data = np.fromfile(f, dtype=dtype)

                # Controllo integrità
                expected_pixels = self.nx * self.ny
                if raw_data.size != expected_pixels:
                    # A volte i file hanno header locali non documentati, proviamo a tagliare
                    if raw_data.size > expected_pixels:
                        raw_data = raw_data[-expected_pixels:]  # Prendi gli ultimi pixel
                    else:
                        print(f"Errore slice {filename}: pixel attesi {expected_pixels}, trovati {raw_data.size}")
                        continue

                # Reshape da striscia a matrice 2D e inserimento nel volume
                volume_data[i, :, :] = raw_data.reshape((self.ny, self.nx))

        # 5. Creazione Oggetto Volume per Vedo
        # Impostiamo lo spacing corretto in mm
        self.volume = vedo.Volume(volume_data, spacing=(self.sz, self.sy, self.sx))

        # Opzionale: A volte Curry inverte l'asse Y o Z.
        # Qui potremmo dover fare self.volume.mirror("y") se le immagini sono sottosopra.

        # Imposta una color map di default (scala di grigi ossea)
        self.volume.mode(1).cmap("gray")
        return True

    def show(self):
        """
        Apre il visualizzatore interattivo a 3 piani (Slicer).
        """
        if self.volume is None:
            print("Nessun volume caricato.")
            return

        print("Avvio visualizzatore 3D...")

        # Slicer3DPlotter è un'applicazione pronta all'uso di Vedo
        # Offre vista Assiale, Sagittale, Coronale e 3D con slider
        plt = Slicer3DPlotter(
            self.volume,
            bg="black",
            bg2="black",
            cmaps=("gray", "bone", "jet", "spectral"),
            use_slider3d=False  # Disabilita slider nel box 3d per pulizia
        )

        # Titolo finestra
        plt.window.SetWindowName(f"Curry Viewer - {self.meta.get('patient_name', 'Unknown')}")
        plt.show().close()