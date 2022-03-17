import sys
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QDoubleValidator
from toggle_stim_ui import Ui_MainWindow

from win_control import get_coords, clicks, clicks_alt


import os

clear = lambda: os.system("cls")


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    stim_is_on = False

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.TurnStimONBtn.clicked.connect(self.turnStimON)
        self.TurnStimOFFBtn.clicked.connect(self.turnStimOFF)
        self.UpdateCoordsBtn.clicked.connect(self.updateCoords)

    def light_on(self):
        self.frame_4.setStyleSheet("background-color: rgb(255, 255, 255)")

    def updateCoords(self):
        get_coords()

    def turnStimON(self):
        print("AAAAA")
        # self.frame_4.setStyleSheet("background-color: rgb(255, 255, 255)")
        if not self.stim_is_on:
            clicks_alt()
            self.stim_is_on = True

        clear()

    def turnStimOFF(self):
        print("AAAAA")
        # self.frame_4.setStyleSheet("background-color: rgb(0, 0, 0)")
        if self.stim_is_on:
            clicks_alt()
            self.stim_is_on = False
        clear()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()

    sys.exit(app.exec_())

