# class to select and visualize HELP pages in a format PDF on web browser
#
from PySide6.QtCore import Signal, QSize
from PySide6.QtWidgets import QDialog, QVBoxLayout, QComboBox, QPushButton
import webbrowser as wb

class MyDialogHelp(QDialog):
    selection_changed = Signal(str)  # Custom signal for selection changes

    def __init__(self):
        super().__init__()
        self.setFixedSize(QSize(260, 150))
        self.setWindowTitle('Help Pages to Show')
        #
        layout = QVBoxLayout()
        self.combo_pages = QComboBox()  # Use lowercase and snake_case for naming consistency
        self.combo_pages.addItems(['select one help page', 'MD Introduction', 'MD Data review', 'MD Data Base',
                                   'MD Tremor',
                                   'MD Upper Limb ROM',
                                   'MD Cervical ROM',
                                   'MD Tug analysis',
                                   'MD Dual task',
                                   'MD Hilbert transform',
                                   'MD Lmd decomposition',
                                   'MD Power spectrum',
                                   'MD Classification',
                                   'MD Ambulatory prj',
                                   'MD IMU description'])
        layout.addWidget(self.combo_pages)
        button_ok = QPushButton('Ok, Close Help')
        layout.addWidget(button_ok)
        self.setLayout(layout)
        #
        self.combo_pages.currentIndexChanged.connect(self.emit_selection_changed)
        button_ok.clicked.connect(self.closedlg)

    def emit_selection_changed(self):
        #
        selected_text = self.combo_pages.currentText()
        print('dialog selected =', selected_text)
        if selected_text == 'MD Introduction':
            pdf_path = 'help_docs/docpdf/ManualMD_intro.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Tremor':
            pdf_path = 'help_docs/docpdf/ManualMD_tremor.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Upper Limb ROM':
            pdf_path = 'help_docs/docpdf/ManualMD_upperLimbRom.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Cervical ROM':
            pdf_path = 'help_docs/docpdf/ManualMD_cervicalROM.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Data Base':
            pdf_path = 'help_docs/docpdf/ManualMD_dbase.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Dual task':
            pdf_path = 'help_docs/docpdf/ManualMD_dualTask.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Hilbert transform':
            pdf_path = 'help_docs/docpdf/ManualMD_HilbertTransform.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Lmd decomposition':
            pdf_path = 'help_docs/docpdf/ManualMD_LMD_decomposition.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Power spectrum':
            pdf_path = 'help_docs/docpdf/ManualMD_power_spectrum.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Data review':
            pdf_path = 'help_docs/docpdf/ManualMD_review.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Tug analysis':
            pdf_path = 'help_docs/docpdf/ManualMD_TUG30s.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Classification':
            pdf_path = 'help_docs/docpdf/ManualMD_analysis_ovr.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD Ambulatory prj':
            pdf_path = 'help_docs/docpdf/MD-AmbulatoryPRJ.pdf'
            wb.open_new(pdf_path)
        if selected_text == 'MD IMU description':
            pdf_path = 'help_docs/docpdf/ManualMD_IMU_description.pdf'
            wb.open_new(pdf_path)
        # close dialog
        self.closedlg()

    def closedlg(self):
        self.close()