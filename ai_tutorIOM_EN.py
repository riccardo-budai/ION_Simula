"""
    Updated version of evaluation of last response and the trend of learning process
"""

# file: ai_tutorIOM_X.py
import sqlite3
from datetime import datetime
import numpy as np


class LearningAgent:
    """
    AI Agent that analyzes user behavior and determines
    the optimal learning path.
    Updated for immediate responsiveness.
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.cur = db_manager.cur

    def log_behavior(self, user_id: int, action_type: str, details: str = "",
                     duration: float = 0.0, session_id: str = None):
        """ Records a behavioral action. """
        try:
            query = """
                INSERT INTO user_behavior_logs (idcode, idsession, timestamp, action_type, details, session_duration)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            self.cur.execute(query, (user_id, session_id, datetime.now(), action_type, details, duration))
            self.db_manager.conn.commit()
        except Exception as e:
            print(f"AI Log Error: {e}")

    def log_ai_feedback(self, user_id: int, session_id: str, trigger: str,
                        recommendation_data: dict):
        """
        Stores the AI recommendation in the database history.
        """
        try:
            query = """
                INSERT INTO ai_feedback_history 
                (idcode, idsession, timestamp, trigger_event, message_text, difficulty_adj, suggested_scenario_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """

            msg = recommendation_data.get('message', '')
            diff = recommendation_data.get('difficulty_adjustment', 'NORMAL')
            scen = recommendation_data.get('suggested_scenario_id', None)

            self.cur.execute(query, (
                user_id,
                session_id,
                datetime.now(),
                trigger,
                msg,
                diff,
                scen
            ))
            self.db_manager.conn.commit()
        except Exception as e:
            print(f"AI Feedback Log Error: {e}")

    def analyze_profile(self, user_id: int):
        """
        Analyzes historical data to build a detailed student profile.
        Returns a dictionary with key metrics including streaks and last outcome.
        """
        metrics = {
            "mastery_score": 0,  # 0-100 (Average of last 10)
            "last_outcome": None,  # True (Correct) / False (Wrong) / None
            "consecutive_correct": 0,  # Streak
            "total_attempts": 0
        }

        try:
            # Retrieve the last 10 attempts ordered by most recent
            self.cur.execute("""
                SELECT esito_corretto, punti_ottenuti, tempo_risposta_sec 
                FROM session_decisions sd
                JOIN sessions s ON sd.idsession = s.idsession
                WHERE s.idcode = ? 
                ORDER BY sd.decision_id DESC 
                LIMIT 10
            """, (user_id,))
            attempts = self.cur.fetchall()

            # [DEBUG] Print what the AI sees in the DB
            print(f"\n[AI ANALYSIS] Found {len(attempts)} recent attempts for User {user_id}")

            if attempts:
                metrics["total_attempts"] = len(attempts)

                # 1. Last Outcome (Record 0 is the most recent thanks to ORDER BY DESC)
                metrics["last_outcome"] = bool(attempts[0][0])

                # 2. Calculate Streak (Consecutive correct answers from the start of the list)
                streak = 0
                for a in attempts:
                    if a[0]:  # If correct
                        streak += 1
                    else:
                        break  # Stop at the first error going backwards in time
                metrics["consecutive_correct"] = streak

                # 3. Mastery Score (Simple weighted average)
                correct_count = sum(1 for a in attempts if a[0])
                metrics["mastery_score"] = int((correct_count / len(attempts)) * 100)

                print(f"[AI ANALYSIS] Last Outcome: {'CORRECT' if metrics['last_outcome'] else 'WRONG'}")
                print(f"[AI ANALYSIS] Streak: {streak} | Mastery (Average): {metrics['mastery_score']}%")

        except Exception as e:
            print(f"AI Error in Performance Analysis: {e}")

        return metrics

    def get_next_recommendation(self, user_id: int):
        """
        Decides the next step based on immediate performance AND long-term mastery.
        """
        profile = self.analyze_profile(user_id)

        mastery = profile["mastery_score"]
        last_correct = profile["last_outcome"]
        streak = profile["consecutive_correct"]

        recommendation = {
            "message": "",
            "suggested_scenario_id": None,
            "difficulty_adjustment": "NORMAL"
        }

        # --- DYNAMIC RECOMMENDATION LOGIC ---TODO verify

        # 1. IMMEDIATE REACTION TO ERROR
        if last_correct is False:
            recommendation["difficulty_adjustment"] = "EASY"
            recommendation[
                "message"] = "I noticed an error in the last response. Let's slow down a bit: I recommend reviewing the basic concepts or retrying with less noise."
            recommendation["suggested_scenario_id"] = "REVIEW_BASICS"
            return recommendation  # Exit immediately

        # 2. NEW USER (Few data points)
        if profile["total_attempts"] < 3:
            recommendation["difficulty_adjustment"] = "NORMAL"
            recommendation["message"] = "Great start! Keep going to calibrate your level."
            return recommendation

        # 3. STREAK MANAGEMENT (Positive Series)
        if streak >= 3:
            recommendation["difficulty_adjustment"] = "INSANE"
            recommendation[
                "message"] = f"{streak} correct answers in a row! You are unstoppable. Activating 'Time Challenge' mode for the next test."
            recommendation["suggested_scenario_id"] = "TIME_CHALLENGE"
            return recommendation

        # 4. AVERAGE-BASED MANAGEMENT (If no extremes present)
        if mastery < 50:
            recommendation["difficulty_adjustment"] = "EASY"
            recommendation[
                "message"] = "Your average suggests practicing with simpler scenarios to consolidate the basics."

        elif 50 <= mastery < 80:
            recommendation["difficulty_adjustment"] = "HARD"
            recommendation[
                "message"] = "You have good competence. Let's try raising the noise level in the next trace."
            recommendation["suggested_scenario_id"] = "SEP_COMPLEX_DRIFT"

        elif mastery >= 80:
            recommendation["difficulty_adjustment"] = "INSANE"
            recommendation["message"] = "Excellent mastery. You are ready for complex surgical scenarios."

        return recommendation