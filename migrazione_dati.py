# file: migrazione_dati.py

import pymysql
import sqlite3
import sys
from datetime import datetime  # [MODIFICA] Aggiunto import per gestire le date

# --- CONFIGURAZIONE ---
# (La tua configurazione è corretta, la lascio invariata)
MARIADB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'db': 'IOM_Simula'
}
SQLITE_FILE = "sqlt3_dbase/IOM_Simula.db"
TABELLE_DA_MIGRARE = [
    'subjects',
    'scenari_setup',
    'sessions',
    'scenari_anomaly',
    'session_decisions'
]


# --- FINE CONFIGURAZIONE ---


def migra_dati():
    """
    Legge i dati da MariaDB e li inserisce in SQLite, preservando gli ID.
    """
    conn_maria = None
    conn_sqlite = None

    try:
        conn_maria = pymysql.connect(**MARIADB_CONFIG)
        conn_sqlite = sqlite3.connect(SQLITE_FILE)
        cur_maria = conn_maria.cursor()
        cur_sqlite = conn_sqlite.cursor()
        cur_sqlite.execute("PRAGMA foreign_keys = ON;")
        print("Connessioni stabilite. Inizio migrazione...")

        for tabella in TABELLE_DA_MIGRARE:
            print(f"--- Processando tabella: {tabella} ---")

            # 1. Leggi i dati da MariaDB
            print(f"  1. Lettura dati da MariaDB...")
            cur_maria.execute(f"SELECT * FROM {tabella};")
            righe = cur_maria.fetchall()

            # Ottieni i nomi delle colonne
            colonne = [desc[0] for desc in cur_maria.description]

            if not righe:
                print(f"  -> Trovate 0 righe. Tabella saltata.")
                continue

            print(f"  -> Trovate {len(righe)} righe.")

            # [MODIFICA] Inizia la sezione di pulizia dati
            print(f"  2. Pulizia dati (gestione NULL e conversione datetime)...")

            righe_pulite = []  # Creiamo una nuova lista per i dati corretti

            # Trova l'indice della colonna 'trigger_delay' (solo se esiste)
            idx_trigger_delay = -1
            if tabella == 'scenari_setup':
                try:
                    idx_trigger_delay = colonne.index('trigger_delay')
                    print(f"  -> Colonna 'trigger_delay' trovata all'indice {idx_trigger_delay}")
                except ValueError:
                    print("  -> ATTENZIONE: colonna 'trigger_delay' non trovata in 'scenari_setup'!")

            # Itera su ogni riga per pulirla
            for riga in righe:
                # Converti la tupla in lista (perché le tuple non si possono modificare)
                riga_modificata = list(riga)

                # [CORREZIONE 1: Errore NOT NULL]
                # Se siamo sulla tabella 'scenari_setup' e la colonna 'trigger_delay' è None...
                if tabella == 'scenari_setup' and idx_trigger_delay != -1:
                    if riga_modificata[idx_trigger_delay] is None:
                        # ...imposta il valore di default a 0
                        riga_modificata[idx_trigger_delay] = 0
                        # print(f"    -> Corretto NULL in 'trigger_delay' per la riga con ID {riga_modificata[0]}") # Debug opzionale

                # [CORREZIONE 2: Avviso DeprecationWarning]
                # Itera su ogni cella della riga e converte i datetime in stringhe
                for i, valore in enumerate(riga_modificata):
                    if isinstance(valore, datetime):
                        riga_modificata[i] = valore.strftime('%Y-%m-%d %H:%M:%S')

                # Aggiungi la riga pulita alla nostra nuova lista
                righe_pulite.append(tuple(riga_modificata))

                # [MODIFICA] Fine sezione pulizia dati

            # 2. Prepara la query di inserimento per SQLite
            segnaposto = ", ".join(["?"] * len(colonne))
            query_insert = f"INSERT INTO {tabella} ({', '.join(colonne)}) VALUES ({segnaposto});"

            print(f"  3. Inserimento dati in SQLite...")

            # 3. Esegui l'inserimento in SQLite USANDO LE RIGHE PULITE
            cur_sqlite.executemany(query_insert, righe_pulite)  # [MODIFICA]

            # 4. Aggiorna la sequenza di AUTOINCREMENT
            colonna_id = colonne[0]
            check_seq = cur_sqlite.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'").fetchone()
            if check_seq:
                cur_sqlite.execute(
                    f"UPDATE sqlite_sequence SET seq = (SELECT MAX({colonna_id}) FROM {tabella}) WHERE name = ?;",
                    (tabella,))
                print(f"  -> Contatore AUTOINCREMENT per '{tabella}' aggiornato.")

            print(f"  -> Dati per '{tabella}' migrati con successo.")

        # Finalizza la transazione
        conn_sqlite.commit()
        print("\n--- MIGRAZIONE COMPLETATA CON SUCCESSO! ---")

    except pymysql.Error as e:
        print(f"\nERRORE MariaDB: {e}", file=sys.stderr)
        if conn_sqlite: conn_sqlite.rollback()
    except sqlite3.Error as e:
        print(f"\nERRORE SQLite: {e}", file=sys.stderr)
        if conn_sqlite: conn_sqlite.rollback()
    except Exception as e:
        print(f"\nERRORE SCONOSCIUTO: {e}", file=sys.stderr)
        if conn_sqlite: conn_sqlite.rollback()
    finally:
        # Chiudi sempre le connessioni
        if cur_maria: cur_maria.close()
        if conn_maria: conn_maria.close()
        if cur_sqlite: cur_sqlite.close()
        if conn_sqlite: conn_sqlite.close()
        print("Connessioni ai database chiuse.")


# --- Esegui lo script ---
if __name__ == "__main__":
    migra_dati()