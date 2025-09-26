from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget
from PyQt6.uic import loadUi
# 这行代码的作用是从 PyQt6 的 uic 模块中导入 loadUi 函数，该函数是用于加载 Qt Designer 生成的 .ui 文件（界面设计文件）的核心工具。
# 假设你用 Qt Designer 设计了一个名为 main.ui 的界面文件，通过 loadUi 可以直接加载该文件并生成对应的界面对象。

class Stats:
    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态创建一个相应的窗口对象
        self.ui = loadUi("D:/qtdesigner/u1.ui", QWidget())  # PyQt6 中loadUi需要指定父对象
        #保存ui文件时候最好保存在自己记得的地方，因为上面要打路径，我这里是绝对路径
        self.ui.pushButton.clicked.connect(self.handleCalc)
        #注意此处pushButton这个名称，跟ui文件中右上角的对象查看器中的名称一一对应，自己命名的时候也要注意
    def handleCalc(self):
        info = self.ui.plainTextEdit.toPlainText()
        #此处plainTextEdit同上
        salary_above_20k = ''
        salary_below_20k = ''
        for line in info.splitlines():
            if not line.strip():
                continue
            parts = line.split(' ')

            parts = [p for p in parts if p]
            name, salary, age = parts
            if int(salary) >= 20000:
                salary_above_20k += name + '\n'
            else:
                salary_below_20k += name + '\n'

        QMessageBox.about(self.ui,
                          '统计结果',
                          f'''薪资20000 以上的有：\n{salary_above_20k}
                    \n薪资20000 以下的有：\n{salary_below_20k}'''
                          )


if __name__ == "__main__":
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec()
#以上代码是在qtdesigner简单设计完框架后，保存界面定义文件，通过python程序加载ui定义，并且动态创建一个相应的窗口对象