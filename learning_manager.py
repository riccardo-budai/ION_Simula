"""
    # file: learning_manager.py
    Management of the learning process
"""

import sqlite3
from datetime import datetime


class LearningManager:
    """
    Manages the logic for recording and retrieving data related
    to the user's learning process.
    """

    def __init__(self, db_manager):
        """
        Initializes the LearningManager with an existing DatabaseManager
        instance to interact with the database.
        """
        if not db_manager or not db_manager.conn:
            raise ValueError("A valid and connected DatabaseManager instance is required.")
        self.db_manager = db_manager
        self.cur = self.db_manager.cur
        self.conn = self.db_manager.conn

        # Stores the start of the learning session
        self.start_time = None
        self.current_session_id = None
        self.current_anomaly_id = None
        print("LearningManager initialized.")

    def start_learning_session(self, session_id: int, anomaly_id: int):
        """
        Starts the timer and stores the current challenge data.
        To be called at the start of every simulation with an anomaly.
        """
        self.start_time = datetime.now()
        self.current_session_id = session_id
        self.current_anomaly_id = anomaly_id
        print(f"Learning session started for session_id={session_id}, anomaly_id={anomaly_id}")

    def record_decision(self, azione_presa: str, esito_corretto: bool, punti: int,
                        valore_parametro: float = None, tempo_risposta_sec: float = None):
        """
        Records the user's decision in the database.
        This is the main function to call from the simulation.
        """
        if not self.current_session_id or not self.current_anomaly_id:
            print("ERROR: Learning session was not started with start_learning_session().")
            return False

        if tempo_risposta_sec is None:
            if not self.start_time:
                print("ERROR: Response time not provided and start_time not available.")
                return False
            # Fallback logic if response_time_sec is not provided
            end_time = datetime.now()
            tempo_risposta = (end_time - self.start_time).total_seconds()
        else:
            tempo_risposta = tempo_risposta_sec

        try:
            # [MOD] Replaced all %s with ? for sqlite3 compatibility
            query = """
                INSERT INTO session_decisions 
                (idsession, anomaly_id, azione_presa, valore_parametro, esito_corretto, punti_ottenuti,
                    tempo_risposta_sec, timestamp_decision)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            data = (
                self.current_session_id,
                self.current_anomaly_id,
                azione_presa,
                valore_parametro,
                esito_corretto,
                punti,
                tempo_risposta
            )
            self.cur.execute(query, data)
            self.conn.commit()
            print(f"Decision recorded successfully! Response time: {tempo_risposta:.2f}s")
            return True
        # [MOD] Specific error handling for sqlite3
        except sqlite3.Error as err:
            print(f"SQL Error while recording decision: {err}")
            self.conn.rollback()
            return False

    def evaluate_user_action(self, user_action: str, expected_action: str, max_points: int):
        """
        Compares the user's action with the expected one and calculates the outcome.
        Returns a tuple (is_correct: bool, points_assigned: int).
        """
        is_correct = (user_action.upper() == expected_action.upper())

        punti_finali = 0
        if is_correct:
            # If correct, assign points (optionally scaled by time)
            punti_finali = max_points
        else:
            # If wrong, 0 points (or negative penalty if preferred)
            punti_finali = 0

        return is_correct, punti_finali

    def check_level_advancement(self, subject_id: int, current_level: int):
        """
        Checks if the user meets the requirements to advance to the next level.
        Criterion: > 80% success in the last 5 simulations of this level.
        """
        try:
            # Select the user's last 5 decisions
            query = """
                SELECT esito_corretto 
                FROM session_decisions sd
                JOIN sessions s ON sd.idsession = s.idsession
                WHERE s.idcode = ? 
                ORDER BY sd.decision_id DESC 
                LIMIT 5
            """
            self.cur.execute(query, (subject_id,))
            results = self.cur.fetchall()

            if len(results) < 5:
                return False, "Insufficient data (at least 5 trials needed)"

            # row[0] is esito_corretto (1 or 0)
            successi = sum([1 for row in results if row[0] == 1])
            percentuale = (successi / 5) * 100

            if percentuale >= 80:
                new_level = current_level + 1
                self._update_subject_level(subject_id, new_level)
                return True, f"Congratulations! Advanced to Level {new_level} (Success: {percentuale}%)"
            else:
                return False, f"Stay at Level {current_level} (Current success: {percentuale}%)"

        except sqlite3.Error as e:
            print(f"Level check error: {e}")
            return False, "DB Error"

    def _update_subject_level(self, subject_id: int, new_level: int):
        """Updates the user's level in the database."""
        try:
            # Assuming the subjects table has a 'level' column
            query = "UPDATE subjects SET level = ? WHERE idcode = ?"
            self.cur.execute(query, (new_level, subject_id))
            self.conn.commit()
            print(f"User {subject_id} promoted to level {new_level}.")
        except sqlite3.Error as e:
            print(f"Unable to update user level: {e}")

    def get_performance_summary(self, subject_id: int, scenario_id: int = None):
        """
        Retrieves a summary of a user's performance.
        If scenario_id is specified, filters results for that scenario.
        """
        try:
            # [MOD] Replaced %s with ?
            query = """
                SELECT 
                    sa.label_id AS anomalia_label,
                    COUNT(sd.decision_id) AS totale_tentativi,
                    SUM(CASE WHEN sd.esito_corretto = 1 THEN 1 ELSE 0 END) AS successi,
                    AVG(sd.tempo_risposta_sec) AS tempo_medio_risposta,
                    SUM(sd.punti_ottenuti) as punteggio_totale
                FROM session_decisions sd
                JOIN sessions s ON sd.idsession = s.idsession
                JOIN scenari_anomaly sa ON sd.anomaly_id = sa.anomaly_id
                WHERE s.idcode = ?
            """
            params = [subject_id]

            if scenario_id:
                # [MOD] Replaced %s with ?
                query += " AND sa.scenario_id = ?"
                params.append(scenario_id)

            query += " GROUP BY sa.label_id ORDER BY anomalia_label;"

            self.cur.execute(query, tuple(params))
            results = self.cur.fetchall()

            # Format results into a list of dictionaries for easy use
            summary = []
            for row in results:
                summary.append({
                    "anomalia": row[0],
                    "tentativi": row[1],
                    "successi": row[2],
                    "percentuale_successo": (row[2] / row[1] * 100) if row[1] > 0 else 0,
                    "tempo_medio": row[3],
                    "punteggio": row[4]
                })
            return summary
        # [MOD] Specific error handling for sqlite3
        except sqlite3.Error as err:
            print(f"SQL Error retrieving performance summary: {err}")
            return []
