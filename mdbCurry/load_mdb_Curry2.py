import subprocess
import pandas as pd
import io
import os

# --- CONFIGURAZIONE ---
db_path = "dataIOM/Casi_Curry.mdb"


def get_data(table_name):
    """Estrae una tabella dal MDB e la converte in DataFrame Pandas"""
    if not os.path.exists(db_path):
        print(f"❌ File non trovato: {db_path}")
        return pd.DataFrame()

    try:
        # mdb-export genera CSV
        res = subprocess.run(["mdb-export", db_path, table_name], capture_output=True, text=True)
        if res.returncode != 0:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(res.stdout))
    except Exception:
        return pd.DataFrame()


# --- LOGICA PRINCIPALE ---
if __name__ == "__main__":
    print("⏳ Caricamento e Unione Dati...")

    # 1. Carica Tabelle
    subjects = get_data("Subjects")
    studies = get_data("Studies")
    links = get_data("StudyInputFileLinks")
    files = get_data("InputFiles")

    if subjects.empty or studies.empty:
        print("❌ Tabelle vuote o non trovate.")
        exit()

    try:
        # 2. MERGE A CASCATA (Gestione Nomi Colonne)

        # Merge A: Soggetti + Studi
        # Se 'CreationDate' è solo in Studi, rimarrà 'CreationDate'.
        # Se c'è conflitto (es. Comment), userà i suffissi.
        merge_1 = pd.merge(subjects, studies, on="SubjectID", suffixes=('_Sogg', '_Stud'))

        # Merge B: + Links
        merge_2 = pd.merge(merge_1, links, on="StudyID")

        # Merge C: + Files
        # QUI ERA L'ERRORE: CreationDate esisteva sia in merge_2 (dagli studi) sia in files.
        # Usiamo suffixes specifici per risolvere il conflitto.
        # Resultato: CreationDate dello studio resta tale (o prende suffisso sx), quella del file diventa _File
        finale = pd.merge(merge_2, files, on="InputFileID", suffixes=('_Stud', '_File'))

        # DEBUG: Stampa tutte le colonne per sicurezza (così vedi come si chiamano)
        print("\n🔍 Colonne disponibili nel database finale:")
        print(finale.columns.tolist())

        # 3. PULIZIA

        # Creiamo il nome file completo
        finale['FullFileName'] = finale.apply(
            lambda row: f"{row['BaseName']}.{row['MainExtension']}" if pd.notna(row['MainExtension']) else row[
                'BaseName'],
            axis=1
        )

        # Cerchiamo la colonna giusta per la data.
        # Pandas applica i suffissi solo se c'è collisione.
        # Controlliamo quale usare.
        colonna_data = 'CreationDate'
        if 'CreationDate_Stud' in finale.columns:
            colonna_data = 'CreationDate_Stud'
        elif 'CreationDate' not in finale.columns and 'CreationDate_x' in finale.columns:
            colonna_data = 'CreationDate_x'  # Fallback se i suffissi falliscono

        print(f"   -> Uso la colonna data: {colonna_data}")

        # Selezione Colonne Finali
        colonne_visibili = [
            'FirstName',
            'LastName',
            colonna_data,  # Usiamo la variabile calcolata sopra
            'Label',
            'FullFileName',
            'FileType'
        ]

        # Ordina
        finale_ordinata = finale[colonne_visibili].sort_values(by=['LastName', colonna_data])

        print("\n✅ RISULTATO (Primi 10 record):")
        print(finale_ordinata.head(10).to_string(index=False))

        # 4. SALVA
        output_csv = "Elenco_File_Pazienti.csv"
        finale_ordinata.to_csv(output_csv, index=False)
        print(f"\n💾 Salvato in: {output_csv}")

    except KeyError as e:
        print(f"\n❌ Errore KeyError: {e}")
        print("La colonna richiesta non esiste. Controlla l'elenco stampato sopra.")
    except Exception as e:
        print(f"\n❌ Errore Generico: {e}")
