from PyQt6.QtWidgets import QApplication, QMainWindow  # 导入QMainWindow
from PyQt6.uic import loadUi

class Stats(QMainWindow):  # 继承QMainWindow
    def __init__(self):
        super().__init__()
        # 用QMainWindow作为基类加载UI
        self.ui = loadUi("D:/qtdesigner/yiyuan.ui", self)  # 直接传self（当前QMainWindow实例）

if __name__ == "__main__":
    app = QApplication([])
    stats = Stats()
    stats.show()
    app.exec()