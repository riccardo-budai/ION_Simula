
import time
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QComboBox


class ChildWindow(QMainWindow):
    # - new window to show text content; correlated to main window
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("read or modifiy AI prompt")
        self.setGeometry(50, 50, 500, 450)  # Set initial position and size
        self.move(550, 50)

        with open("res/styleSheets/styleSheetApp.qss", "r") as f:
            self.setStyleSheet(f.read())

        '''
        # self.setStyleSheet("background-color: gray;")
        self.setStyleSheet("QWidget { font-size: 14px; };")
        # Crea e applica la palette
        palette = create_high_contrast_palette(background_color, text_color, highlight_color)
        self.setPalette(palette)
        # Crea e applica il foglio di stile
        stylesheet = create_high_contrast_stylesheet(background_color, text_color, highlight_color)
        self.setStyleSheet(stylesheet)
        '''

        # Create a central widget to hold the layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # Main vertical layout for the child window
        main_layout = QVBoxLayout(central_widget)
        #
        # add 3 buttons on bottom of the window: Edit, Save, Exit
        layoutButtons = QHBoxLayout()
        #
        self.editText = QPushButton("Edit text")
        self.editText.setAutoFillBackground(True)
        #
        self.saveText = QPushButton("Save")
        self.saveText.setEnabled(False)
        #
        self.exitButton = QPushButton("Exit")
        layoutButtons.addWidget(self.editText)
        layoutButtons.addWidget(self.saveText)
        layoutButtons.addWidget(self.exitButton)
        self.exitButton.clicked.connect(self.closeWin)
        self.saveText.clicked.connect(self.saveTextprompt)
        self.editText.clicked.connect(self.editTextprompt)
        #
        self.lblFName = QLabel('')
        # palette = self.lblFName.palette()
        # palette.setColor(QPalette.ColorRole.Button, QColor("lightgreen"))  # Use QColor for more options
        # self.lblFName.setPalette(palette)
        self.lblFName.setStyleSheet("border: 1px solid lightgreen;")
        layoutFilename = QHBoxLayout()

        self.editFname = QComboBox()     # QPushButton('Overwrite filename')
        self.editFname.addItem('Manage file...')
        self.editFname.addItem('Overwrite file...')
        self.editFname.addItem('Rename file...')
        self.editFname.addItem('Update database')
        self.editFname.setCurrentIndex(0)
        self.editFname.activated.connect(self.editFilePrompt)
        layoutFilename.addWidget(self.lblFName)
        layoutFilename.addWidget(self.editFname)
        #
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        # self.setCentralWidget(self.text_edit)

        # Add widgets and layouts to the main layout
        main_layout.addWidget(self.text_edit)
        main_layout.addLayout(layoutFilename)
        main_layout.addLayout(layoutButtons)

    def editTextprompt(self):
        # - edit and modify content of prompt previously saved
        self.text_edit.setReadOnly(False)
        self.text_edit.append(time.strftime('%d/%m/%Y %H:%M:%S\n'))
        self.saveText.setEnabled(True)

    def editFilePrompt(self):
        if self.editFname.currentIndex() == 1:
            print('overwrite file !')

    def saveTextprompt(self):
        # - obtain text info from self.text_edit
        info = self.text_edit.toPlainText()
        # - overwrite the previous file .txt
        with open('prompts/' + self.lblFName.text(), "w") as file:
            file.write(info)
        self.saveText.setEnabled(False)

    def closeWin(self):
        self.close()