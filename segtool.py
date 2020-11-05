import utility
import sys
import cv2
import matrix_processing
import image_processing

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import config

# File Name global variale
RESEARCH_BASE_DIR = config.RESEARCH_BASE_DIR
IMAGE_NAME = "Image/Interior/interior7/interior7_0.jpg"
OUTPUT_FILE = RESEARCH_BASE_DIR + '/' + utility.add_name(IMAGE_NAME.split("/")[-1], "_divided")
SEG_FILE_NAME = RESEARCH_BASE_DIR + '/' + utility.add_name(IMAGE_NAME.split("/")[-1], "", extension="bin")
SEG_SAVE_NAME = RESEARCH_BASE_DIR + '/' + utility.add_name(IMAGE_NAME.split("/")[-1], "_userInput", extension="bin")
# Constant
CHANGE_DIVIED = "Image/example/temp.jpg"

# Init Global Data for classify segmentation.
totalClass = [[]]
nowIndex = 0
eraseMode = False
eraseList = []
[divided_class, class_number, class_total, class_border] = utility.load_result(SEG_FILE_NAME)

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Segment GUI Helper")
        grid = QGridLayout()
        
        # Add Image ( Segmentation )
        self.imageLabel = ImageLabel(self.clickHandler, OUTPUT_FILE)

        grid.addWidget(self.imageLabel, 0, 0)
        
        # Add Button List ( 3개 버튼 - before / erase / next )
        grid.addWidget(self.buttonList(), 1, 0)

        # Add Save Button
        saveButton = QPushButton(self)
        saveButton.setText("Save")
        grid.addWidget(saveButton, 2, 0)
        saveButton.clicked.connect(self.saveData)
        grid.setRowStretch(0, 20)

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

    def clickHandler(self, event):
        now = (event.pos().x(), event.pos().y())
        if eraseMode:
            eraseList.append(now)
        else:
            totalClass[nowIndex].append(now)
        self.update()
    
    def saveData(self):
        global nowIndex     # 현재 추가하고 있는 index
        global divided_class    # Class Number map
        global class_number     # Class Number 의 종류
        global class_total      # 각 Class들의 total Coords
        global class_border     # Class border.
        
        img = cv2.imread(IMAGE_NAME)
        (height, width, _) = img.shape

        class_total, class_number, divided_class = mergeGroup(class_total, class_number, divided_class, nowIndex)
        utility.save_result([divided_class, class_number, class_total, class_border], SEG_SAVE_NAME)
        class_count = [len(class_total[i]) for i in range(len(class_total))]
        class_color = image_processing.get_class_color(utility.read_image(IMAGE_NAME), class_total, class_count)
        dc_image = utility.divided_class_into_image(divided_class, class_number, class_color, width, height, class_number)
        utility.save_image(dc_image, CHANGE_DIVIED)
        self.imageLabel.changePixmap(CHANGE_DIVIED)

        # 저장하고 화면 Refresh 까지 진행해야 함.

class ImageLabel(QLabel):
    def __init__(self, clickHandler, fileName, parent=None):
        QLabel.__init__(self, parent)
        self.pixmap = QPixmap(fileName)
        self.setPixmap(self.pixmap)
        self._whenClicked = clickHandler
    
    def paintEvent(self, event):
        qp = QPainter(self)
        qp.drawPixmap(self.rect(), self.pixmap)

        # Drawing Painter.
        for now in eraseList:
            drawErasePoint(qp, now)
        for now in totalClass[nowIndex]:
            drawAddPoint(qp, now)
    
    def mouseReleaseEvent(self, event):
        self._whenClicked(event)
    
    def changePixmap(self, fileName):
        self.pixmap = QPixmap(fileName)
        self.setPixmap(self.pixmap)

def drawErasePoint(qPaint, point):
    pen = QPen(Qt.red, 1)
    qPaint.setPen(pen)
    for ct in class_total:
        if point in ct:
            for coord in ct:
                qPaint.drawPoint(coord[0], coord[1])

def drawAddPoint(qPaint, point):
    pen = QPen(Qt.blue, 1)
    qPaint.setPen(pen)
    for ct in class_total:
        if point in ct:
            for coord in ct:
                qPaint.drawPoint(coord[0], coord[1])

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

def eraseToggle():
    global eraseMode
    eraseMode = not eraseMode

def mergeGroup(classTotal, classNumber, dividedClass, nowIndex):
    # totalClass[nowIndex] 의 좌표들이 있는 모든 classTotal을 합쳐서 return.
    # 해당 영역이 써있는 dividedClass와 classNumber는 자동으로 제일 첫 Area로 바꿔서 저장한다.
    mergeIndex = []
    global totalClass
    for nowCoord in totalClass[nowIndex]:
        indx = findIndex(classTotal, nowCoord)
        if indx not in mergeIndex:
            mergeIndex.append(indx)
    mergeIndex.sort()
    if len(mergeIndex) > 1:
        retClassTotal = []
        retClassNumber = []

        for i in range(0, len(classTotal)):
            if i not in mergeIndex or i == mergeIndex[0]:
                retClassTotal.append(classTotal[i])
                retClassNumber.append(classNumber[i])

        for i in range(1, len(mergeIndex)):
            retClassTotal[mergeIndex[0]] += classTotal[mergeIndex[i]]
        
        addIndex = mergeIndex[0]
        matrix_processing.set_area(dividedClass, retClassTotal[addIndex], retClassNumber[addIndex])
    else:
        retClassTotal = [classTotal[i] for i in range(len(classTotal))]
        retClassNumber = [classNumber[i] for i in range(len(classNumber))]
    totalClass = [[]]
    
    global eraseList
    if len(eraseList) != 0:
        for el in eraseList:
            indx = findIndex(retClassTotal, el)
            matrix_processing.set_area(dividedClass, retClassTotal[indx], 0)
            if indx != -1:
                del retClassTotal[indx]
                del retClassNumber[indx]
        eraseList = []
    return retClassTotal, retClassNumber, dividedClass
    
def findIndex(classTotal, coord):
    for ct in range(0, len(classTotal)):
        if coord in classTotal[ct]:
            return ct
    return -1
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())