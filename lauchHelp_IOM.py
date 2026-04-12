import sys
import os
import subprocess
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox


class FinestraHelp(QWidget):
    def __init__(self):
        super().__init__()

        # --- Impostazioni della Finestra ---
        self.setWindowTitle("IOM Simula - Controller")
        self.setGeometry(100, 100, 300, 150)  # Posizione x, y, larghezza, altezza

        # --- Layout ---
        layout = QVBoxLayout()

        # --- Il Bottone ---
        self.btn_help = QPushButton("📚 Apri Guida Interattiva")
        # Personalizziamo un po' lo stile del bottone (Opzionale)
        self.btn_help.setStyleSheet("""
            QPushButton {
                background-color: #0E1117; 
                color: white; 
                font-size: 14px; 
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #262730;
            }
        """)

        # Colleghiamo il click del bottone alla funzione (Slot)
        self.btn_help.clicked.connect(self.avvia_streamlit)

        # Aggiungiamo il bottone al layout
        layout.addWidget(self.btn_help)
        self.setLayout(layout)

    def avvia_streamlit(self):
        """
        Lancia lo script Streamlit in un processo separato.
        """
        # 1. Troviamo il percorso assoluto del file app_helpIOM.py
        # Questo serve perché se lanci il programma da un'altra cartella, Python deve sapere dove cercare.
        cartella_corrente = os.path.dirname(os.path.abspath(__file__))
        path_script_streamlit = os.path.join(cartella_corrente, "app_helpIOM.py")

        # Verifica di sicurezza: il file esiste?
        if not os.path.exists(path_script_streamlit):
            QMessageBox.critical(self, "Errore", f"Non trovo il file:\n{path_script_streamlit}")
            return

        print(f"Avvio Streamlit da: {path_script_streamlit}")

        # 2. Lanciamo il comando
        # sys.executable garantisce che usiamo lo stesso Python dell'ambiente virtuale corrente
        try:
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", path_script_streamlit])
        except Exception as e:
            QMessageBox.critical(self, "Errore di avvio", f"Impossibile avviare Streamlit.\nErrore: {str(e)}")


# --- Blocco di avvio dell'applicazione ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    finestra = FinestraHelp()
    finestra.show()

    sys.exit(app.exec())