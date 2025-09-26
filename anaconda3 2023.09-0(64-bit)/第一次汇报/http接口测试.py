from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.uic import loadUi
class Stats:
    def __init__(self):
        self.ui = loadUi("D:/qtdesigner/HTTP.ui", QWidget())
if __name__ == "__main__":
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec()


