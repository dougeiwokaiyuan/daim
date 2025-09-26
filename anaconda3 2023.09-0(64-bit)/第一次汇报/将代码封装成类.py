from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton,  QPlainTextEdit,QMessageBox
import sys
class Stats:
    def __init__(self):
        # self指代当前类的实例对象本身
        # 这样，每个Stats实例（如stats1 = Stats()、stats2 = Stats()）都会有自己独立的window和textEdit，彼此互不干扰。
        self.window = QMainWindow()
        # 注意区分Stats.window = QMainWindow()是给类本身定义类属性，而非给实例定义属性
        # 此时创建多个Stats实例（如stats1 = Stats()、stats2 = Stats()）
        # 它们会共用同一个window、textEdit和button，修改其中一个实例的控件状态会影响所有实例，这显然不符合 “每个窗口独立运行” 的需求。
        self.window.resize(500, 400)
        self.window.move(300, 300)
        self.window.setWindowTitle('薪资统计')

        self.textEdit = QPlainTextEdit(self.window)
        self.textEdit.setPlaceholderText("请输入薪资表")
        self.textEdit.move(10, 25)
        self.textEdit.resize(300, 350)

        self.button = QPushButton('统计', self.window)
        self.button.move(380, 80)

        self.button.clicked.connect(self.handleCalc)

    def handleCalc(self):
        # 从文本编辑框中获取输入的所有文本信息
        info = self.textEdit.toPlainText()

        # 初始化两个字符串变量，分别用于存储薪资20000以上和以下的人员姓名
        salary_above_20k = ''
        salary_below_20k = ''

        # 按行分割文本信息，逐行处理
        for line in info.splitlines():
            # 跳过空行（去除前后空格后为空的行）
            if not line.strip():
                continue

            # 按空格分割每行内容，得到各个字段
            parts = line.split(' ')
            # 过滤掉分割后列表中的空字符串（处理多个空格的情况）
            parts = [p for p in parts if p]

            # 提取姓名、薪资和年龄信息（假设输入格式为"姓名 薪资 年龄"）
            name, salary, age = parts

            # 判断薪资是否达到20000，将姓名添加到对应分组
            if int(salary) >= 20000:
                salary_above_20k += name + '\n'  # 换行符使每个姓名单独一行
            else:
                salary_below_20k += name + '\n'

        # 弹出消息框显示统计结果
        QMessageBox.about(
            self.window,  # 父窗口
            '统计结果',  # 消息框标题
            # 消息内容，使用f-string格式化字符串
            f'''薪资20000 以上的有：\n{salary_above_20k}
            \n薪资20000 以下的有：\n{salary_below_20k}'''
        )


app = QApplication(sys.argv)
stats = Stats()
stats.window.show()
app.exec()
# 如果不使用类，就需要通过全局变量或函数参数来传递这些状态，如果代码变得复杂大量的话会让代码变得复杂且难以追踪。
# 可以将类想成一个箱子，将需要的都装在一起，便于维护。
# 如果想生成两个窗口的话，就可以生成两个箱子，互不干扰