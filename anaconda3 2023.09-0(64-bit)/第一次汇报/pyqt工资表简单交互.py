from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton,  QPlainTextEdit,QMessageBox
# 导入了 PyQt6 中用于创建 GUI 的核心组件：
# QApplication：应用程序的主对象，管理应用程序的生命周期。
# QMainWindow：主窗口控件，作为程序的顶层窗口。
# QPushButton：按钮控件，用于触发操作（如 “统计” 按钮）。
# QPlainTextEdit：纯文本编辑控件，用于用户输入薪资信息。
# QMessageBox：消息框控件，用于显示统计结果。
import sys
def handleCalc():#信号与槽：在 Qt 系统中， 当界面上一个控件被操作时，比如 被点击、被输入文本、被鼠标拖拽等， 就会发出 信号 ，英文叫 signal 。就是表明一个事件（比如被点击、被输入文本）发生了。

#我们可以预先在代码中指定 处理这个 signal 的函数，这个处理 signal 的函数 叫做 slot 。
    info = textEdit.toPlainText()#让用户输入和编辑多行文本，并把信息赋值给info

    # 薪资20000 以上 和 以下 的人员名单
    salary_above_20k = ''#先初始化空字符串，后续填入信息
    salary_below_20k = ''
    for line in info.splitlines():
        if not line.strip():#去除字符串首尾的空白字符，例如空格、制表符、换行符等，如果这一行进行去除操作后是为空，则进行下一行的操作
            continue
        parts = line.split(' ')#单个空格字符将字符串 line 分割成多个子字符串，并返回一个包含这些子字符串的列表
        # 去掉列表中的空字符串内容
        parts = [p for p in parts if p]
        name,salary,age = parts
        if int(salary) >= 20000:
            salary_above_20k += name + '\n'
        else:
            salary_below_20k += name + '\n'

    QMessageBox.about(window,
                '统计结果',
                f'''薪资20000 以上的有：\n{salary_above_20k}
                \n薪资20000 以下的有：\n{salary_below_20k}'''
                )
#以上交互部分总结起来就是：先让用户填入信息，再处理信息里的无用信息，提取有效信息并输出

app = QApplication(sys.argv)

window = QMainWindow()
window.resize(500, 400)
window.move(300, 300)
window.setWindowTitle('薪资统计')

textEdit = QPlainTextEdit(window)
# 此处window是父对象，而textEdit是子对象。
# 令 “textEdit 是 window 的一部分”，这种关系让界面控件能有序地组合在一起，而不是零散地漂浮在屏幕上。
# 显示归属：textEdit 会显示在 window 内部，而不是独立窗口。关闭 window 时，textEdit 也会跟着关闭（父对象销毁时，子对象自动销毁）。
# 坐标参考：textEdit.move(10, 25) 中的坐标（10,25）是相对于 window 的左上角计算的，而不是整个屏幕。
# 事件传递：如果 textEdit 没处理某些事件（比如鼠标点击），会自动传递给 window 处理。
textEdit.setPlaceholderText("请输入薪资表")
textEdit.move(10,25)
textEdit.resize(300,350)

button = QPushButton('统计', window)#此处也是父子关系
button.move(380,80)
button.clicked.connect(handleCalc)
# 此处是信号与槽的关联关系
# button.clicked 是按钮的 “信号”（signal）—— 当按钮被点击时，会发出这个信号（告诉程序 “我被点了”）。
# handleCalc 是 “槽函数”（slot）—— 专门处理这个信号的函数（收到 “被点击” 的信号后，执行统计逻辑）。
# connect 方法的作用就是把 “信号” 和 “槽函数” 绑在一起，让按钮被点击时，自动触发 handleCalc 函数的执行。
window.show()#显示窗口

app.exec()
# 有了这行代码，程序会 “卡住” 在这里（进入循环），不断等待和处理各种事件（用户操作、系统通知等）。
# 直到用户关闭窗口，这个循环才会结束，程序才会真正退出。