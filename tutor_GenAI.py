import sys
import google.generativeai as genai
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                               QWidget, QTextEdit, QLineEdit, QPushButton)
from PySide6.QtCore import Qt

# Configura la tua API KEY qui
CHIAVE_API = 'AIzaSyCd0zvMqCrVv4LPP15F__5RcZOQfV2h1Tw'

genai.configure(api_key=CHIAVE_API)


class TutorAIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tutor AI - Neurofisiologia Intraoperatoria")
        self.resize(600, 500)

        # 1. IMPOSTAZIONE DELLA SCHERMATA (GUI)
        layout = QVBoxLayout()

        # Area dove leggeremo la conversazione (sola lettura)
        self.area_chat = QTextEdit()
        self.area_chat.setReadOnly(True)
        layout.addWidget(self.area_chat)

        # Casella di testo dove lo studente scrive
        self.input_studente = QLineEdit()
        self.input_studente.setPlaceholderText("Scrivi qui la tua risposta...")
        # Permette di inviare il messaggio premendo "Invio"
        self.input_studente.returnPressed.connect(self.invia_messaggio)
        layout.addWidget(self.input_studente)

        # Bottone "Invia"
        self.bottone_invia = QPushButton("Invia")
        self.bottone_invia.clicked.connect(self.invia_messaggio)
        layout.addWidget(self.bottone_invia)

        widget_centrale = QWidget()
        widget_centrale.setLayout(layout)
        self.setCentralWidget(widget_centrale)

        # 2. INIZIALIZZAZIONE DELL'AGENTE AI (Gemini)
        self.modello = genai.GenerativeModel('gemini-2.5-pro')
        self.chat = self.modello.start_chat(history=[])

        # 3. GESTIONE DEGLI STATI
        # 0 = Valutazione, 1 = Pianificazione, 2 = Verifica
        self.stato_attuale = 0
        self.argomento = "Monitoraggio dei Potenziali Evocati Motori (MEP)"

        # Avviamo la prima interazione
        self.avvia_valutazione()

    def aggiungi_testo_chat(self, mittente, testo):
        """Funzione di supporto per aggiungere testo formattato alla chat."""
        if mittente == "Tutor":
            self.area_chat.append(f"<b><font color='blue'>Tutor AI:</font></b> {testo}<br>")
        else:
            self.area_chat.append(f"<b><font color='green'>Tu:</font></b> {testo}<br>")

    def avvia_valutazione(self):
        """Fase 1: Facciamo la prima domanda per capire il livello."""
        self.aggiungi_testo_chat("Tutor", f"Benvenuto! Iniziamo il modulo su: {self.argomento}. "
                                          f"Sto preparando una domanda di valutazione...")

        prompt = f"""Sei un tutor di neurofisiologia intraoperatoria. 
        Fai UNA SOLA domanda aperta allo studente per valutare la sua conoscenza 
        sul tema: '{self.argomento}'. Sii professionale ma incoraggiante."""

        risposta_ai = self.chat.send_message(prompt)
        self.aggiungi_testo_chat("Tutor", risposta_ai.text)

    def invia_messaggio(self):
        """Gestisce cosa succede quando lo studente preme 'Invia'."""
        testo_studente = self.input_studente.text()
        if not testo_studente.strip():  # Evita di inviare messaggi vuoti
            return

        # Mostriamo il messaggio dello studente nella GUI e puliamo la casella di input
        self.aggiungi_testo_chat("Studente", testo_studente)
        self.input_studente.clear()

        # In base allo "stato" attuale, decidiamo come l'AI deve rispondere
        if self.stato_attuale == 0:
            # Siamo nella fase di valutazione, ora passiamo al piano di studio
            self.genera_piano_studio(testo_studente)
            self.stato_attuale = 1

        elif self.stato_attuale == 1:
            # Siamo nella fase di verifica continua
            self.continua_lezione(testo_studente)

    def genera_piano_studio(self, risposta_studente):
        """Fase 2: Valuta la risposta e crea il piano (Syllabus)."""
        prompt = f"""Lo studente ha risposto alla domanda di valutazione in questo modo: '{risposta_studente}'.
        1. Dimmi se il suo livello è Principiante, Intermedio o Esperto.
        2. Proponi un breve piano di studio a tappe (3 step) per migliorare su '{self.argomento}'.
        3. Concludi chiedendo se è pronto per iniziare il primo step."""

        risposta_ai = self.chat.send_message(prompt)
        self.aggiungi_testo_chat("Tutor", risposta_ai.text)

    def continua_lezione(self, risposta_studente):
        """Fase 3: Lezione e verifica continua."""
        prompt = f"""Rispondi allo studente: '{risposta_studente}'.
        Spiega il concetto successivo del piano di studio e fagli una domanda 
        mirata per verificare se ha capito. Usa un linguaggio tecnico appropriato 
        per la neurofisiologia intraoperatoria."""

        risposta_ai = self.chat.send_message(prompt)
        self.aggiungi_testo_chat("Tutor", risposta_ai.text)


# --- AVVIO DELL'APP PYSIDE6 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    finestra = TutorAIWindow()
    finestra.show()
    sys.exit(app.exec())