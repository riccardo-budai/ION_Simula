# 'database_managerIOM_sqlt3.py'

import sqlite3
from datetime import datetime
import threading
from data_modelsIOM_sqlt3 import Subject, Session, Scenari, Anomaly


class DatabaseManager:
    """
        # [MODIFICA] Accesso al database SQLite3
    """

    # [MODIFICA] Il costruttore ora accetta un percorso di file
    def __init__(self, db_path: str):
        self.conn = None
        self.cur = None
        self.lock = threading.Lock()
        try:
            # [MODIFICA] Connessione al file database SQLite
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.cur = self.conn.cursor()

            # [MODIFICA] Abilita il supporto per le chiavi esterne (FOREIGN KEY)
            self.cur.execute("PRAGMA foreign_keys = ON;")

            print(f"Connesso al database SQLite '{db_path}' con successo.")

        # [MODIFICA] Gestione degli errori specifica di SQLite
        except sqlite3.Error as err:
            print(f"FATAL: Errore di connessione al database SQLite: {err}")
            raise err

        self.initialize_tables()

    def initialize_tables(self):
        """
        Esegue gli script SQL per creare le tabelle di setup e risultati.
        """

        # [MODIFICA] Sintassi SQL adattata per SQLite
        SQL_CREATE_SCENARI_SETUP = """
            CREATE TABLE IF NOT EXISTS scenari_setup (
                scenario_id INTEGER PRIMARY KEY AUTOINCREMENT,
                label_id TEXT NOT NULL,
                nome_scenario TEXT NOT NULL,
                descrizione TEXT,
                trigger_tipo TEXT NOT NULL,
                trigger_delay INTEGER NOT NULL,
                data_creazione TEXT DEFAULT (datetime('now','localtime'))
            )
            """

        # [MODIFICA] Sintassi SQL adattata per SQLite
        # [CORREZIONE] Campo 'anomaly' rinominato 'anomaly_id' per coerenza con la dataclass
        # [NOTA] Hai aggiunto 'azione_corretta', che ora è inclusa.
        SQL_CREATE_ANOMALY_SETUP = """
            CREATE TABLE IF NOT EXISTS scenari_anomaly (
                anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                scenario_id INTEGER NOT NULL,
                label_id TEXT NOT NULL,
                descrizione TEXT,
                json_anomaly_id TEXT NOT NULL,
                data_creazione TEXT DEFAULT (datetime('now','localtime')),
                azione_corretta  TEXT DEFAULT ('N/A'),
                FOREIGN KEY (scenario_id) REFERENCES scenari_setup(scenario_id)
        )
        """

        # [MODIFICA] Sintassi SQL adattata per SQLite (BOOLEAN -> INTEGER, DECIMAL -> REAL)
        SQL_CREATE_SESSION_DECISIONS = """
            CREATE TABLE IF NOT EXISTS session_decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                idsession INTEGER NOT NULL,
                anomaly_id INTEGER NOT NULL,
                timestamp_decision TEXT DEFAULT (datetime('now','localtime')),
                azione_presa TEXT NOT NULL,
                valore_parametro REAL, 
                esito_corretto INTEGER NOT NULL,
                punti_ottenuti INTEGER DEFAULT 0,
                tempo_risposta_sec REAL, 
                FOREIGN KEY (idsession) REFERENCES sessions(idsession),
                FOREIGN KEY (anomaly_id) REFERENCES scenari_anomaly(anomaly_id)
        )
        """

        query_behavior = """
            CREATE TABLE IF NOT EXISTS user_behavior_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                idcode INTEGER,
                idsession INTEGER,
                timestamp DATETIME,
                action_type TEXT,       -- 'LOGIN', 'SIMULATION_START', 'HELP_READ', 'IDLE_TIMEOUT'
                details TEXT,           -- Es. 'Opened PDF Tutorial', 'Simulation duration 50s'
                session_duration REAL   -- Tempo in secondi (se applicabile)
            );
            """

        query_ai_history = """
            CREATE TABLE IF NOT EXISTS ai_feedback_history (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                idcode INTEGER,
                idsession INTEGER,
                timestamp DATETIME,
                trigger_event TEXT,        -- Es: 'SIMULATION_END', 'LOGIN_CHECK'
                message_text TEXT,         -- Il testo mostrato all'utente
                difficulty_adj TEXT,       -- 'HARD', 'EASY', 'NORMAL'
                suggested_scenario_id TEXT -- Se l'AI ha proposto uno scenario specifico
            );
            """


        if not self.conn:
            print("Errore: connessione al database non attiva.")
            return

        # [MODIFICA] Gestione degli errori specifica di SQLite
        try:
            self.cur.execute(SQL_CREATE_SCENARI_SETUP)
            print("Tabella 'scenari_setup' verificata/creata con successo.")

            # [CORREZIONE] Creava anche 'subjects' e 'sessions',
            # ma erano mancanti. Li aggiungo per completezza.

            SQL_CREATE_SUBJECTS = """
            CREATE TABLE IF NOT EXISTS subjects (
                idcode INTEGER PRIMARY KEY AUTOINCREMENT,
                firstname TEXT,
                lastname TEXT,
                height INTEGER,
                weight INTEGER,
                age INTEGER,
                gender TEXT,
                fname TEXT,
                dbaseclass TEXT,
                level TEXT,
                datetime TEXT
            )
            """
            self.cur.execute(SQL_CREATE_SUBJECTS)
            print("Tabella 'subjects' verificata/creata con successo.")

            SQL_CREATE_SESSIONS = """
            CREATE TABLE IF NOT EXISTS sessions (
                idcode INTEGER,
                idsession INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT,
                taskname TEXT,
                simulacode TEXT,
                sesnote TEXT,
                fname TEXT,
                dbasename TEXT,
                datetime TEXT,
                FOREIGN KEY (idcode) REFERENCES subjects (idcode)
            )
            """
            self.cur.execute(SQL_CREATE_SESSIONS)
            print("Tabella 'sessions' verificata/creata con successo.")

            self.cur.execute(SQL_CREATE_ANOMALY_SETUP)
            print("Tabella 'scenari_anomaly' verificata/creata con successo.")

            self.cur.execute(SQL_CREATE_SESSION_DECISIONS)
            print("Tabella 'session_decisions' verificata/creata con successo.")

            self.cur.execute(query_behavior)
            print("Tabella user_behaviour_logs verificata/creata consuccesso")

            self.cur.execute(query_ai_history)
            print("Tabella query_ai_history verificata/creata consuccesso")

            self.conn.commit()

        except sqlite3.Error as err:
            print(f"Errore SQL durante l'inizializzazione delle tabelle: {err}")
            self.conn.rollback()

    def get_all_scenarios(self):
        """Recupera tutti gli scenari dal database, ordinati per nome."""
        try:
            with self.lock:
                query = """
                    SELECT scenario_id, label_id, nome_scenario, descrizione, 
                           trigger_tipo, trigger_delay, data_creazione
                    FROM scenari_setup
                    ORDER BY nome_scenario ASC;
                """
                self.cur.execute(query)
                results = self.cur.fetchall()
            return [Scenari(*row) for row in results]
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nel recuperare gli scenari: {err}")
            return []

    def get_subject_by_id(self, subject_id: int):
        """Recupera un singolo soggetto dato il suo ID."""
        try:
            with self.lock:
                query = """
                            SELECT idcode, firstname, lastname, height, weight, age, 
                                   gender, fname, dbaseclass, level, datetime 
                            FROM subjects WHERE idcode = ?;
                        """
                # [MODIFICA] Segnaposto da %s a ?
                self.cur.execute(query, (subject_id,))
                result = self.cur.fetchone()
            return Subject(*result) if result else None
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nel recuperare il soggetto {subject_id}: {err}")
            return None

    def get_all_subjects(self):
        """Recupera tutti i soggetti dal database, ordinati per nome."""
        try:
            with self.lock:
                query = """
                            SELECT idcode, firstname, lastname, height, weight, age, 
                                   gender, fname, dbaseclass, level, datetime
                            FROM subjects 
                            ORDER BY firstname ASC;
                        """
                self.cur.execute(query)
                results = self.cur.fetchall()
            return [Subject(*row) for row in results]
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nel recuperare i soggetti: {err}")
            return []

    def get_sessions_for_subject(self, subject_id: int):
        """Recupera tutte le sessioni per un dato ID soggetto."""
        try:
            # [MODIFICA] Segnaposto da %s a ?
            query = "SELECT * FROM sessions WHERE idcode = ? ORDER BY datetime DESC;"
            self.cur.execute(query, (subject_id,))
            results = self.cur.fetchall()
            return [Session(*row) for row in results]
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nel recuperare le sessioni: {err}")
            return []

    def update_subject(self, subject_data: dict):
        """Aggiorna un record soggetto."""
        try:
            # [MODIFICA] Segnaposto da %s a ?
            query = """UPDATE subjects SET 
                   firstname=?, lastname=?, age=?, 
                   gender=?, dbaseclass=?, fname=?, level=? 
                   WHERE idcode=?;"""
            data = (
                subject_data['firstname'],
                subject_data['lastname'],
                subject_data['age'],
                subject_data['gender'],
                subject_data['dbaseclass'],
                subject_data['fname'],
                subject_data['level'],
                subject_data['idcode']
            )
            self.cur.execute(query, data)
            self.conn.commit()
            print(f"Soggetto {subject_data['idcode']} aggiornato.")
            return True
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nell'aggiornare il soggetto: {err}")
            self.conn.rollback()
            return False

    def add_new_subject(self, subject_data: dict):
        """Aggiunge un nuovo soggetto e restituisce il suo nuovo ID."""
        try:
            with self.lock:
                # [MODIFICA] Segnaposto da %s a ?
                query = """INSERT INTO subjects 
                           (firstname, lastname, age, gender,
                            fname, dbaseclass, datetime, level) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?);"""

                # [CORREZIONE] Gestione flessibile del datetime
                dt_value = subject_data['datetime']
                if isinstance(dt_value, datetime):
                    # Se è un oggetto datetime, formattalo in stringa
                    dt_str = dt_value.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Altrimenti, assumi sia già una stringa (o None) e usalo
                    dt_str = str(dt_value)

                data = (
                    subject_data['firstname'], subject_data['lastname'],
                    subject_data['age'], subject_data['gender'],
                    subject_data['fname'], subject_data['dbaseclass'], dt_str,  # [MODIFICA] Usiamo la stringa sicura
                    subject_data['level']
                )

                self.cur.execute(query, data)
                self.conn.commit()
                new_id = self.cur.lastrowid
            print(f"Nuovo soggetto aggiunto con ID: {new_id}")
            return new_id
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nell'aggiungere il soggetto: {err}")
            self.conn.rollback()
            return None

    def update_session(self, session_data: dict):
        """Aggiorna un record sessione."""
        try:
            with self.lock:
                # [MODIFICA] Segnaposto da %s a ?
                query = """UPDATE sessions SET 
                           fname=?, taskname=?, session=?, simulacode=?, 
                           dbasename=?, sesnote=?, datetime=? 
                           WHERE idsession=?;"""

                # [CORREZIONE] Gestione flessibile del datetime (CAUSA DELL'ERRORE)
                dt_value = session_data['datetime']
                if isinstance(dt_value, datetime):
                    # Se è un oggetto datetime, formattalo in stringa
                    dt_str = dt_value.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Se è GIA' una stringa (letta dal DB), usala così com'è
                    dt_str = str(dt_value)

                data = (
                    session_data['fname'], session_data['taskname'], session_data['session'],
                    session_data['simulacode'], session_data['dbasename'], session_data['sesnote'],
                    dt_str,  # [MODIFICA] Usiamo la stringa sicura
                    session_data['idsession']
                )
                self.cur.execute(query, data)
                self.conn.commit()
            print(f"Sessione {session_data['idsession']} aggiornata.")
            return True
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nell'aggiornare la sessione: {err}")
            self.conn.rollback()
            return False

    def add_new_session(self, session_data: dict):
        """Aggiunge una nuova sessione e restituisce il suo nuovo ID."""
        try:
            with self.lock:
                # [MODIFICA] Segnaposto da %s a ?
                query = """INSERT INTO sessions 
                           (idcode, fname, taskname, session, simulacode, dbasename, sesnote, datetime) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?);"""

                # [CORREZIONE] Gestione flessibile del datetime
                dt_value = session_data['datetime']
                if isinstance(dt_value, datetime):
                    # Se è un oggetto datetime, formattalo in stringa
                    dt_str = dt_value.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Altrimenti, assumi sia già una stringa (o None) e usalo
                    dt_str = str(dt_value)


                data = (
                    session_data['idcode'], session_data['fname'], session_data['taskname'],
                    session_data['session'], session_data['simulacode'], session_data['dbasename'],
                    session_data['sesnote'], dt_str  # [MODIFICA] Usiamo la stringa sicura
                )
                self.cur.execute(query, data)
                self.conn.commit()
                new_id = self.cur.lastrowid
            print(f"Nuova sessione aggiunta con ID: {new_id}")
            return new_id
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nell'aggiungere la sessione: {err}")
            self.conn.rollback()
            return None

    def get_anomalies_for_scenario(self, scenario_id: int):
        """Recupera tutte le anomalie per un dato ID scenario."""
        try:
            with self.lock:
                # [NOTA] La tua query ora include 'azione_corretta', che è corretto
                # se hai aggiornato anche la dataclass Anomaly.
                query = """
                    SELECT anomaly_id, scenario_id, label_id, descrizione, 
                           json_anomaly_id, data_creazione, azione_corretta
                    FROM scenari_anomaly
                    WHERE scenario_id = ?
                    ORDER BY label_id ASC;
                """
                self.cur.execute(query, (scenario_id,))
            results = self.cur.fetchall()
            return [Anomaly(*row) for row in results]
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nel recuperare le anomalie per lo scenario {scenario_id}: {err}")
            return []

    def add_anomaly(self, anomaly_data: dict):
        """Aggiunge un nuovo record anomalia e restituisce il suo ID."""
        try:
            with self.lock:
                # [MODIFICA] Segnaposto da %s a ?
                # [NOTA] Se vuoi salvare anche 'azione_corretta' qui, devi aggiungerla
                # sia alla query che alla tupla 'data'.
                query = """INSERT INTO scenari_anomaly 
                           (scenario_id, label_id, descrizione, json_anomaly_id) 
                           VALUES (?, ?, ?, ?);"""
                data = (
                    anomaly_data['scenario_id'], anomaly_data['label_id'],
                    anomaly_data['descrizione'], anomaly_data['json_anomaly_id']
                    # Se 'azione_corretta' deve essere aggiunta qui:
                    # anomaly_data['azione_corretta']
                )
                self.cur.execute(query, data)
                self.conn.commit()
            new_id = self.cur.lastrowid
            print(f"Nuova anomalia aggiunta con ID: {new_id}")
            return new_id
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nell'aggiungere l'anomalia: {err}")
            self.conn.rollback()
            return None

    def get_anomaly_json_filename_by_label(self, anomaly_label_id: str):
        """
        Recupera il nome del file JSON (json_anomaly_id) da scenari_anomaly
        utilizzando il label_id fornito dalla sessione (simulacode).
        """
        try:
            with self.lock:
                # [MODIFICA] Segnaposto da %s a ?
                query = """
                    SELECT json_anomaly_id
                    FROM scenari_anomaly
                    WHERE label_id = ?
                    LIMIT 1;
                """
                self.cur.execute(query, (anomaly_label_id,))
            result = self.cur.fetchone()
            return result[0] if result else None
        # [MODIFICA] Gestione errori SQLite
        except sqlite3.Error as err:
            print(f"Errore nel recuperare il nome del file JSON per label '{anomaly_label_id}': {err}")
            return None

    def close(self):
        """Chiude la connessione al database in modo sicuro."""
        if self.cur: self.cur.close()
        if self.conn: self.conn.close()
        print("Connessione al database SQLite chiusa.")


if __name__ == "__main__":
    import os
    DB_PATH = "sqlt3_dbase/IOM_Simula.db"

    # Assicurati che la cartella "sqlt3_dbase" esista
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    print(f"Avvio test: creo il database in {DB_PATH}")
    app = DatabaseManager(DB_PATH)
    print("Test completato. Database e tabelle create.")