import utility
import sys
import cv2

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

FILE_NAME = "Image/example/chair1-divided.jpg"
SEG_FILE_NAME = "Image/example/chair1.bin"
SEG_SAVE_NAME = "Image/example/chair1_userInput.bin"

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Segment GUI Helper")
        grid = QGridLayout()
        
        # Add Image ( Segmentation )
        imageLabel = ImageLabel(clickHandler, FILE_NAME)

        grid.addWidget(imageLabel, 0, 0)
        
        # Add Button List ( 3개 버튼 - before / erase / next )
        grid.addWidget(self.buttonList(), 1, 0)

        # Add Save Button
        saveButton = QPushButton(self)
        saveButton.setText("Save")
        grid.addWidget(saveButton, 2, 0)
        grid.setRowStretch(0, 20)

        img = cv2.imread(FILE_NAME)
        (height, width, _) = img.shape

        self.setLayout(grid)
        self.center()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def buttonList(self):
        buttonBox = QGroupBox()
        gridBox = QGridLayout()
        beforeClassBtn = QPushButton(self)
        beforeClassBtn.setText("Before Class")
        beforeClassBtn.clicked.connect(beforeClass)
        gridBox.addWidget(beforeClassBtn, 0, 0)

        eraseModebtn = QPushButton(self)
        eraseModebtn.setText("Erase Mode")
        eraseModebtn.clicked.connect(eraseToggle)
        gridBox.addWidget(eraseModebtn, 0, 1)

        nextClassBtn = QPushButton(self)
        nextClassBtn.setText("Next Class")
        nextClassBtn.clicked.connect(afterClass)
        gridBox.addWidget(nextClassBtn, 0, 2)

        buttonBox.setLayout(gridBox)
        return buttonBox

class ImageLabel(QLabel):
    def __init__(self, clickHandler, fileName, parent=None):
        QLabel.__init__(self, parent)
        pixmap = QPixmap(fileName)
        self.setPixmap(pixmap)
        self._whenClicked = clickHandler
    
    def mouseReleaseEvent(self, event):
        self._whenClicked(event)

totalClass = [[]]
nowIndex = 0

def clickHandler(event):
    global nowIndex
    global totalClass
    totalClass[nowIndex].append((event.pos().x(), event.pos().y()))
    print(totalClass)

def beforeClass():
    global nowIndex
    global totalClass
    if nowIndex != 0:
        nowIndex -= 1

def afterClass():
    global nowIndex
    global totalClass
    if len(totalClass[nowIndex]) != 0:
        totalClass.append([])
        nowIndex += 1

eraseMode = False
def eraseToggle():
    global eraseMode
    eraseMode = not eraseMode

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())