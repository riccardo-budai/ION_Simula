

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QTextEdit
from PySide6.QtCore import Qt

########################################################################################################################
class ReportSepWindow(QWidget):
    """
    Finestra dedicata alla visualizzazione dei tempi di registrazione
    (log) e di altre informazioni per la generazione di report.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Report e Log dei Tempi di Registrazione")
        self.setGeometry(1055, 200, 500, 700)
        self.setWindowFlag(Qt.WindowType.Window)

        # Layout Principale
        main_layout = QVBoxLayout(self)

        # Tab Widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self._waterfall_log_buffer = []

        # --- Pagina 1: Log Waterfall (Tracce Complete) ---
        self.waterfall_log_text = QTextEdit()
        self.waterfall_log_text.setReadOnly(True)
        self.tab_widget.addTab(self.waterfall_log_text, "Log Waterfall (Tracce a N Avg)")

        # Inserimento del titolo iniziale
        self.waterfall_log_text.append(f"LOG WATERFALL TRACCE\n{'=' * 30}\n")

        # --- Pagina 2: Log Completo (Tutti i Blocchi Elaborati) ---
        self.full_log_text = QTextEdit()
        self.full_log_text.setReadOnly(True)
        self.tab_widget.addTab(self.full_log_text, "Log Completo (Blocchi di Averaging)")

        # Inserimento del titolo iniziale
        self.full_log_text.append(f"LOG COMPLETO DEI BLOCCHI DI AVERAGING\n{'=' * 40}\n")

    def _update_waterfall_display(self):
        """ Ricarica il QTextEdit dal buffer, garantendo l'ordine e il titolo. """
        title_line = "LOG WATERFALL TRACCE"
        separator_line = "=" * 30

        # Unisce tutti gli elementi del buffer
        log_content = "\n".join(self._waterfall_log_buffer)

        # Ricostruisce il testo (il più recente è in alto)
        full_text = f"{title_line}\n{separator_line}\n{log_content}"
        self.waterfall_log_text.setText(full_text)

    def add_waterfall_entry(self, log_entry: str):
        """ Aggiunge una riga di log per ogni traccia aggiunta al waterfall (average completo). """
        # La riga è formattata qui, in modo che sia pronta per la visualizzazione
        new_entry = f"[{log_entry}] {self.parent().TRACE_KEY} avg {self.parent().N_AVERAGES}"

        # Aggiunge il nuovo log all'inizio del buffer (FIFO visualizzato come LIFO)
        self._waterfall_log_buffer.insert(0, new_entry)

        # Aggiorna la visualizzazione
        self._update_waterfall_display()

    def remove_oldest_waterfall_entry(self):
        """ Rimuove la riga di log più vecchia (l'ultima nel buffer). """
        if self._waterfall_log_buffer:
            # Rimuove l'ultimo elemento (quello più vecchio, aggiunto per ultimo)
            self._waterfall_log_buffer.pop()
            # Aggiorna la visualizzazione
            self._update_waterfall_display()

    def add_full_log_entry(self, log_entry: str):
        """ Aggiunge una riga di log per OGNI blocco elaborato dal worker. """
        # Aggiunge in basso, come un log standard, e fa scorrere (append)
        self.full_log_text.append(f"[{log_entry}] Elaborazione blocco {self.parent().main_averages_blocks + 1}")

    # Funzioni di utilità per i report (futura espansione)
    def get_waterfall_log(self) -> str:
        """ Ritorna il log del waterfall per la generazione di report. """
        return self.waterfall_log_text.toPlainText()

    def get_full_log(self) -> str:
        """ Ritorna il log completo per la generazione di report. """
        return self.full_log_text.toPlainText()

