import sqlite3
from datetime import datetime

class LearningManager:
    """
    Manages the logic for recording and retrieving data related
    to the user's learning process.
    """

    def __init__(self, db_manager):
        if not db_manager or not db_manager.conn:
            raise ValueError("A valid and connected DatabaseManager instance is required.")
        self.db_manager = db_manager
        self.cur = self.db_manager.cur
        self.conn = self.db_manager.conn

        # Stores the start of the learning session
        self.start_time = None
        self.current_session_id = None
        self.current_anomaly_id = None
        print("LearningManager initialized (using tables from DatabaseManager).")

    def start_learning_session(self, session_id: str, anomaly_id: int):
        self.start_time = datetime.now()
        self.current_session_id = session_id
        self.current_anomaly_id = anomaly_id
        print(f"Learning session started for session_id={session_id}, anomaly_id={anomaly_id}")

    def record_decision(self, azione_presa: str, esito_corretto: bool, punti: int,
                        valore_parametro: float = None, tempo_risposta_sec: float = None):
        """
        Registra la decisione. Sincronizzato con le colonne di DatabaseManager.
        """
        if not self.current_session_id:
            print("ERROR: Learning session was not started. current_session_id is None.")
            return False

        anom_id = self.current_anomaly_id if self.current_anomaly_id else 0

        if tempo_risposta_sec is None:
            if not self.start_time:
                tempo_risposta = 0.0
            else:
                end_time = datetime.now()
                tempo_risposta = (end_time - self.start_time).total_seconds()
        else:
            tempo_risposta = tempo_risposta_sec

        try:
            # [CORREZIONE] Uso 'timestamp_decision' come definito nel tuo DatabaseManager
            query = """
                INSERT INTO session_decisions 
                (idsession, anomaly_id, azione_presa, valore_parametro, esito_corretto, punti_ottenuti, tempo_risposta_sec, timestamp_decision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            data = (
                self.current_session_id,
                anom_id,
                azione_presa,
                valore_parametro,
                1 if esito_corretto else 0,
                punti,
                tempo_risposta,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            self.cur.execute(query, data)
            self.conn.commit()
            print(f"[DB SUCCESS] Decision recorded! Time: {tempo_risposta:.2f}s, Score: {punti}")
            return True
        except sqlite3.Error as err:
            print(f"[DB ERROR] SQL Error while recording decision: {err}")
            self.conn.rollback()
            return False

    def get_performance_summary(self, subject_id: str, scenario_id: int = None):
        try:
            query = """
                SELECT 
                    sa.label_id AS anomalia_label,
                    COUNT(sd.decision_id) AS totale_tentativi,
                    SUM(CASE WHEN sd.esito_corretto = 1 THEN 1 ELSE 0 END) AS successi,
                    AVG(sd.tempo_risposta_sec) AS tempo_medio_risposta,
                    SUM(sd.punti_ottenuti) as punteggio_totale
                FROM session_decisions sd
                JOIN sessions s ON sd.idsession = s.idsession
                LEFT JOIN scenari_anomaly sa ON sd.anomaly_id = sa.anomaly_id
                WHERE s.idcode = ?
            """
            params = [subject_id]
            if scenario_id:
                query += " AND sa.scenario_id = ?"
                params.append(scenario_id)

            query += " GROUP BY sa.label_id ORDER BY anomalia_label;"

            self.cur.execute(query, tuple(params))
            results = self.cur.fetchall()

            summary = []
            for row in results:
                lbl = row[0] if row[0] else "Unknown/Manual"
                summary.append({
                    "anomalia": lbl,
                    "tentativi": row[1],
                    "successi": row[2],
                    "percentuale_successo": (row[2] / row[1] * 100) if row[1] > 0 else 0,
                    "tempo_medio": row[3] if row[3] else 0.0,
                    "punteggio": row[4] if row[4] else 0
                })
            return summary
        except sqlite3.Error as err:
            print(f"SQL Error retrieving performance summary: {err}")
            return []