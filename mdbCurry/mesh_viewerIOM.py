import os
import re
import numpy as np
import vedo


class CurryMeshViewer:
    def __init__(self):
        self.actors = []  # Lista degli oggetti 3D da mostrare
        self.descriptions = []  # Legenda

    def load_file(self, filepath):
        """
        Carica un file mesh o sorgente e lo aggiunge alla scena.
        """
        if not os.path.exists(filepath):
            return False

        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        print(f"Tentativo caricamento: {filename} ({ext})")

        try:
            # --- CASO A: SORGENTI / DIPOLI (.rs3, .cdr) ---
            if ext in ['.rs3', '.cdr']:
                return self._parse_dipoles(filepath, filename)

            # --- CASO B: SUPERFICI ASCII CURRY (.vcd, .bo0, .s00...s99) ---
            is_curry = False
            if ext in ['.vcd', '.bo0', '.bo1', '.bo2', '.bt1', '.bt2']:
                is_curry = True
            elif re.match(r'^\.s\d{2}$', ext):
                is_curry = True

            if is_curry:
                print(f"  -> Rilevato formato Curry Surface per {filename}")
                return self.parse_curry_surface_file(filepath)

            # --- CASO C: MESH STANDARD (.pom, .stl, ecc) ---
            else:
                print(f"  -> Rilevato formato Standard per {filename}")
                return self._load_surface(filepath, filename)

        except Exception as e:
            print(f"Errore critico nel caricamento di {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _parse_dipoles(self, filepath, label):
        """Legge i file di testo Curry (.rs3) per estrarre posizioni."""
        positions = []
        vectors = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vals = [float(p) for p in parts if p.replace('.', '', 1).replace('-', '', 1).isdigit()]
                        if len(vals) >= 3:
                            positions.append(vals[:3])
                            if len(vals) >= 6:
                                vectors.append(vals[3:6])
                            else:
                                vectors.append([0, 0, 1])
                    except ValueError:
                        continue

            if not positions:
                print(f"Nessun dipolo valido trovato in {label}")
                return False

            pts = vedo.Points(positions, r=10, c="red")
            pts.name = label
            self.actors.append(pts)

            if vectors:
                start_pts = np.array(positions)
                end_pts = start_pts + (np.array(vectors) * 10)
                arrows = vedo.Arrows(start_pts, end_pts, c="yellow")
                self.actors.append(arrows)

            self.descriptions.append(f"🔴 {label} ({len(positions)} dipoli)")
            return True

        except Exception as e:
            print(f"Errore parsing dipoli {label}: {e}")
            return False

    def parse_curry_surface_file(self, file_path):
        """
        Legge un file di superficie Curry (.vcd/.bo0/.s0x) e estrae anche la DESCRIZIONE.
        """
        filename = os.path.basename(file_path)
        locations = []
        triangles = []

        # ### NUOVO: Lista per salvare le righe di commento
        description_lines = []

        current_block = None

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                u_line = line.upper()

                # --- Rilevamento Blocchi ---
                if "LOCATION_LIST" in u_line and ("START" in u_line or "=" in u_line or "LIST" in u_line):
                    current_block = "LOCATIONS"
                    continue
                elif "TRIANGLE_LIST" in u_line and ("START" in u_line or "=" in u_line or "LIST" in u_line):
                    current_block = "TRIANGLES"
                    continue
                elif "NORMALS_LIST" in u_line:
                    current_block = "NORMALS"
                    continue

                # ### NUOVO: Rilevamento inizio Descrizione
                elif "POINT_DESCRIPTION" in u_line and ("START" in u_line or "LIST" in u_line):
                    current_block = "DESCRIPTION"
                    continue

                # Gestione fine blocco
                if "END" in u_line:
                    # Se trova END nudo o END_LIST, chiude il blocco corrente
                    current_block = None
                    continue

                if "COMPRESSED_LIST" in u_line:
                    print(f"⚠️ Attenzione: {filename} contiene dati compressi non supportati.")
                    return False

                # --- Parsing Dati ---
                if current_block == "LOCATIONS":
                    if '=' in line: continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            coords = [float(p) for p in parts[:3]]
                            locations.append(coords)
                        except ValueError:
                            pass

                elif current_block == "TRIANGLES":
                    if '=' in line: continue
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            indices = [int(p) for p in parts[:3]]
                            triangles.append(indices)
                        except ValueError:
                            pass

                # ### NUOVO: Cattura testo descrizione
                elif current_block == "DESCRIPTION":
                    # Ignoriamo righe vuote o simboli strani se necessario
                    if len(line) > 1:
                        description_lines.append(line)

            # --- Preparazione testo Descrizione ---
            # Uniamo le righe trovate. Usiamo "\n" per andare a capo nella legenda o " " per una riga sola.
            # Qui uso "\n   " per indentarlo visivamente sotto il titolo
            extra_info = ""
            if description_lines:
                extra_info = "\n   " + "\n   ".join(description_lines)

            # --- CONTROLLO DATI ---
            if len(locations) == 0:
                print(f"❌ Nessun punto trovato in {filename}.")
                return False

            verts = np.array(locations)
            mesh = None

            # --- Logica Colori ---
            color = "gold"
            alpha = 0.5
            fn_lower = filename.lower()
            if ".s00" in fn_lower:
                color = "flesh";
                alpha = 0.3
            elif ".s01" in fn_lower:
                color = "grey";
                alpha = 0.4
            elif ".bo" in fn_lower:
                color = "lightblue";
                alpha = 0.2

            # --- CREAZIONE OGGETTO GRAFICO ---
            if len(triangles) > 0:
                faces = np.array(triangles)
                if faces.min() > 0:
                    faces = faces - 1

                mesh = vedo.Mesh([verts, faces])
                mesh.c(color).alpha(alpha)
                try:
                    mesh.lw(0.1)
                except:
                    pass
                mesh.compute_normals()

                # ### MODIFICATO: Aggiunge extra_info alla descrizione
                self.descriptions.append(f"🕸️ {filename} (Mesh: {len(faces)} pol.){extra_info}")

            else:
                print(f"⚠️ {filename}: Triangoli non trovati. Visualizzo come Nuvola di Punti.")
                mesh = vedo.Points(verts, r=4)
                mesh.c(color).alpha(0.8)

                # ### MODIFICATO: Aggiunge extra_info alla descrizione
                self.descriptions.append(f"🔵 {filename} (Cloud: {len(locations)} pts){extra_info}")

            mesh.name = filename
            self.actors.append(mesh)
            return True

        except Exception as e:
            print(f"Errore parsing surface {filename}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_surface(self, filepath, label):
        try:
            mesh = vedo.load(filepath)
            if mesh is None: return False
            mesh.c("grey").alpha(0.3)
            mesh.name = label
            if ".pom" in label:
                if not hasattr(mesh, 'ncells') or mesh.ncells == 0:
                    mesh = vedo.Points(mesh.vertices, r=6, c="blue")
            self.actors.append(mesh)
            self.descriptions.append(f"🕸️ {label}")
            return True
        except Exception as e:
            print(f"Errore load generico {label}: {e}")
            return False

    def show(self):
        if not self.actors:
            print("Nessun attore da mostrare.")
            return

        plt = vedo.Plotter(title="Curry Mesh Viewer", axes=1)

        if self.descriptions:
            desc_text = "\n".join(self.descriptions)
            # Aumento un po' la grandezza della legenda per far leggere il testo extra
            legend = vedo.Text2D(desc_text, pos="top-left", s=0.7, bg="yellow", alpha=0.2, font="Calco")
            plt.add(legend)

        plt.show(self.actors, viewup="z").close()
