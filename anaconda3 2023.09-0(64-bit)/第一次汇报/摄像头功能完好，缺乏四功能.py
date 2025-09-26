from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QFileDialog
from PyQt6.uic import loadUi
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import cv2
import os
import time
import pymysql
from threading import Thread
from PyQt6.QtCore import pyqtSignal
# 关闭YOLO调试信息
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO


# 共享实例类
class SI:
    loginWin = None
    mainWin = None


# 1. 登录验证线程
class LoginThread(QThread):
    result_signal = pyqtSignal(bool, str)

    def __init__(self, username, password):
        super().__init__()
        self.username = username
        self.password = password

        # 数据库连接配置
        self.db_config = {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "576499183",
            "database": "detection_system",
            "charset": "utf8mb4"
        }

    def run(self):
        print("子线程：开始执行数据库查询...")
        try:
            # 连接数据库
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor(pymysql.cursors.DictCursor)

            # 查询SQL
            sql = """
                SELECT * 
                FROM user 
                WHERE account = %s 
                  AND password = %s
            """
            cursor.execute(sql, (self.username, self.password))
            result = cursor.fetchone()

            # 验证逻辑
            if result:
                self.result_signal.emit(True, "登录成功")
            else:
                self.result_signal.emit(False, "账号或密码错误")

            # 关闭连接
            cursor.close()
            conn.close()

        except pymysql.MySQLError as e:
            print(f"子线程：数据库错误: {str(e)}")
            self.result_signal.emit(False, f"数据库错误：{str(e)}")
        except Exception as e:
            print(f"子线程：未知异常: {str(e)}")
            self.result_signal.emit(False, f"未知错误：{str(e)}")


# 2. 登录窗口类
class Win_Login(QWidget):
    def __init__(self):
        super().__init__()
        try:
            # 加载UI文件
            self.ui = loadUi("D:/qtdesigner/jiance.ui", self)
            print("登录窗口：UI文件加载成功")
        except Exception as e:
            print(f"登录窗口：UI加载失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"UI文件加载失败：{str(e)}")
            return

        try:
            # 绑定事件
            self.ui.sign_in.clicked.connect(self.onSignIn)
            self.ui.enter_the_password.returnPressed.connect(self.onSignIn)
            print("登录窗口：事件绑定成功")
        except AttributeError as e:
            print(f"登录窗口：控件不存在: {str(e)}")
            QMessageBox.critical(self, "错误", f"控件不存在：{str(e)}")

    def onSignIn(self):
        print("\n登录按钮被点击，开始处理...")
        try:
            # 获取用户输入
            username = self.ui.lineEdit.text().strip()
            password = self.ui.enter_the_password.text().strip()
            print(f"输入的用户名: {username}, 密码: {password}")

            # 简单校验
            if not username or not password:
                QMessageBox.warning(self, "输入错误", "用户名和密码不能为空！")
                return

            # 创建并启动登录子线程
            self.login_thread = LoginThread(username, password)
            self.login_thread.result_signal.connect(self.handle_login_result)
            self.login_thread.start()
            print("子线程已启动，等待结果...")

        except Exception as e:
            print(f"onSignIn方法出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"处理登录时出错：{str(e)}")

    def handle_login_result(self, success, msg):
        print(f"\n收到登录结果：成功={success}, 消息={msg}")
        if success:
            QMessageBox.information(self, "成功", msg)
            try:
                SI.mainWin = Win_Main()
                SI.mainWin.show()
                self.enter_the_password.setText('')
                self.hide()
            except Exception as e:
                print(f"打开主窗口失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"打开主窗口失败：{str(e)}")
        else:
            QMessageBox.warning(self, "失败", msg)


# 3. 主窗口类（包含YOLO功能）
class Win_Main(QWidget):
    update_detected_signal = pyqtSignal(QtGui.QImage)  # 传递QImage
    update_log_signal = pyqtSignal(str)                # 传递日志文本
    def __init__(self):
        super().__init__()
        self.update_detected_signal.connect(self.update_detected_image)
        self.update_log_signal.connect(self.append_log)
        #待停止状态标记（避免延迟期间重复操作或继续处理帧）
        self.is_stopping = False  # False=正常运行，True=已点击停止，等待1秒后结束
        # 尝试加载UI文件，如果失败则使用代码创建UI
        self.use_designer_ui = True
        try:
            self.ui = loadUi("D:/qtdesigner/main11111.ui", self)
            print("主窗口：UI加载成功")
            self.setupDesignerUI()
        except Exception as e:
            print(f"主窗口：UI加载失败，将使用代码创建UI: {str(e)}")
            self.use_designer_ui = False
            self.setupCodeUI()

        # 初始化YOLO相关组件
        self.initYOLOComponents()

        # 绑定按钮事件
        self.bindEvents()

    def setupDesignerUI(self):
        """从设计器UI中获取控件"""
        try:
            self.label_ori_video = self.ui.label_ori_video
            self.label_treated = self.ui.label_treated
            self.textLog = self.ui.textLog
            self.videoBtn = self.ui.videoBtn
            self.camBtn = self.ui.camBtn
            self.stopBtn = self.ui.stopBtn
            # self.logoutBtn = self.ui.logoutBtn  # 确保UI中有此按钮

            # 设置视频显示区域样式
            self.label_ori_video.setMinimumSize(520, 400)
            self.label_treated.setMinimumSize(520, 400)
            self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
            self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')
        except AttributeError as e:
            print(f"设置设计器UI控件失败: {str(e)}")
            QMessageBox.warning(self, "警告", f"UI控件不完整：{str(e)}")

    def update_detected_image(self, q_img):
        """主线程中更新检测画面（安全操作UI）"""
        self.label_treated.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def append_log(self, text):
        """主线程中添加日志（安全操作UI）"""
        self.textLog.append(text)
    def initYOLOComponents(self):
        """初始化YOLO相关组件"""
        # 定义定时器，控制视频显示帧率
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        # 加载YOLO模型
        try:
            self.model = YOLO('yolov8n.pt')
            self.textLog.append("✅ YOLO模型加载成功")
        except Exception as e:
            self.textLog.append(f"❌ YOLO模型加载失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"YOLO模型加载失败：{str(e)}")

        # 视频帧处理队列
        self.frameToAnalyze = []

        # 启动帧分析线程
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

        # 视频捕获对象
        self.cap = None

    def bindEvents(self):
        """绑定按钮事件"""
        self.camBtn.clicked.connect(self.startCamera)
        self.stopBtn.clicked.connect(self.stop)
        self.videoBtn.clicked.connect(self.openVideoFile)
        # self.logoutBtn.clicked.connect(self.logout)

    def startCamera(self):
        """启动摄像头（增加详细日志和异常捕获）"""
        try:
            self.update_log_signal.emit("ℹ️ 尝试启动摄像头...")  # 日志：开始启动

            # 释放旧资源
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.update_log_signal.emit("ℹ️ 已释放旧摄像头资源")

            # 尝试打开摄像头（分步骤检查）
            self.update_log_signal.emit("ℹ️ 正在打开摄像头设备...")
            self.cap = cv2.VideoCapture(0)  # 基础写法，兼容多系统

            # 检查设备是否打开
            if not self.cap.isOpened():
                self.update_log_signal.emit("❌ 摄像头设备无法打开（可能被占用或不存在）")
                return

            # 尝试读取一帧（验证摄像头是否正常工作）
            ret, test_frame = self.cap.read()
            if not ret:
                self.update_log_signal.emit("❌ 摄像头无法读取帧（设备故障或格式不支持）")
                self.cap.release()
                self.cap = None
                return

            self.update_log_signal.emit("✅ 摄像头启动成功，开始显示画面")
            if not self.timer_camera.isActive():
                self.timer_camera.start(50)  # 启动定时器

        except Exception as e:
            # 捕获所有异常并详细输出
            err_msg = f"❌ 摄像头启动过程出错：{str(e)}（类型：{type(e).__name__}）"
            self.update_log_signal.emit(err_msg)
            print(f"【摄像头启动异常】{err_msg}")  # 控制台也打印
            # 出错后清理资源
            if hasattr(self, 'cap'):
                self.cap.release()
                self.cap = None

    def openVideoFile(self):
        """打开本地视频文件"""
        # 释放可能存在的旧资源
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

        # 打开文件选择对话框
        filePath, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if not filePath:
            return

        # 打开视频文件
        self.cap = cv2.VideoCapture(filePath)
        if not self.cap.isOpened():
            self.textLog.append(f"❌ 视频文件打开失败：{filePath}")
            QMessageBox.warning(self, "警告", f"视频文件打开失败：{filePath}")
            return

        self.textLog.append(f"✅ 视频文件打开成功：{os.path.basename(filePath)}")
        if not self.timer_camera.isActive():
            self.timer_camera.start(30)  # 约33fps


    def show_camera(self):
        try:
            # 若已进入停止流程，直接返回（不读取新帧）
            if self.is_stopping:
                return

            # 原有逻辑：检查cap是否存在
            if not hasattr(self, 'cap') or self.cap is None:
                self.update_log_signal.emit("⚠️ 无摄像头资源，停止读取帧")
                self.stop()  # 此时调用stop会触发延迟，不会重复执行
                return

            ret, frame = self.cap.read()
            if not ret:
                self.update_log_signal.emit("⚠️ 无法读取帧（视频流结束）")
                self.stop()
                return

            # 原有帧处理逻辑...
            frame = cv2.resize(frame, (520, 400))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 显示原始画面
            q_img = QtGui.QImage(
                frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0],
                QtGui.QImage.Format.Format_RGB888
            )
            self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(q_img))

            # 帧入队前检查是否已进入停止流程
            if self.is_stopping:
                return  # 不加入新帧
            if len(self.frameToAnalyze) < 2:
                self.frameToAnalyze.append(frame_rgb)
                self.update_log_signal.emit(f"ℹ️ 帧加入队列，当前队列长度：{len(self.frameToAnalyze)}")
            else:
                self.update_log_signal.emit(f"⚠️ 队列已满（{len(self.frameToAnalyze)}帧），丢弃当前帧")

        except Exception as e:
            # 若已进入停止流程，不输出错误日志
            if not self.is_stopping:
                err_msg = f"❌ 帧读取出错：{str(e)}"
                self.update_log_signal.emit(err_msg)
                print(f"【帧读取异常】{err_msg}")
            self.stop()

    def frameAnalyzeThreadFunc(self):
        """帧分析线程：延迟期间不处理帧"""
        while True:
            # 若已进入停止流程，短暂休眠后继续循环（避免CPU占用）
            if self.is_stopping:
                time.sleep(0.1)
                continue

            # 原有逻辑：检查队列是否为空
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue

            try:
                frame = self.frameToAnalyze.pop(0)
                results = self.model(frame)[0]
                img_with_boxes = results.plot(line_width=1)

                # 转换为QImage并更新界面
                q_img = QtGui.QImage(
                    img_with_boxes.data,
                    img_with_boxes.shape[1],
                    img_with_boxes.shape[0],
                    QtGui.QImage.Format.Format_RGB888
                )
                self.update_detected_signal.emit(q_img)

                # 输出检测日志前检查是否已停止
                if not self.is_stopping:
                    detected_classes = [results.names[int(cls)] for cls in results.boxes.cls]
                    if detected_classes:
                        self.update_log_signal.emit(f"🔍 检测到：{', '.join(set(detected_classes))}")

            except Exception as e:
                # 若已进入停止流程，不输出错误日志
                if not self.is_stopping:
                    self.update_log_signal.emit(f"❌ 帧处理错误：{str(e)}")
                    print(f"帧分析线程异常：{e}")


    def stop(self):
        """停止视频流和分析：延迟1秒后执行真正的资源释放"""
        # 1. 若已进入停止流程（避免重复点击停止按钮），直接返回
        if self.is_stopping:
            return

        # 2. 标记为“待停止状态”，禁止后续帧处理和日志输出
        self.is_stopping = True
        self.update_log_signal.emit("⏳ 已触发停止，将在1秒后结束...")  # 告知用户延迟

        # 3. 延迟1秒后，调用真正的停止逻辑（用singleShot实现非阻塞延迟）
        QtCore.QTimer.singleShot(1000, self._real_stop)  # 1000ms=1秒

    def _real_stop(self):
        """真正的停止逻辑：1秒后执行资源释放和界面清理"""
        # 停止定时器（不再触发帧读取）
        if self.timer_camera.isActive():
            self.timer_camera.stop()

        # 释放摄像头/视频资源
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

        # 清空显示区
        self.label_ori_video.clear()
        self.label_treated.clear()

        # 清空帧队列（避免延迟期间残留的帧继续触发分析）
        if hasattr(self, 'frameToAnalyze'):
            self.frameToAnalyze.clear()

        # 输出最终停止日志
        self.update_log_signal.emit("🛑 已停止视频处理（延迟1秒完成）")

        # 重置“待停止状态”（便于后续再次启动）
        self.is_stopping = False

    def logout(self):
        """退出登录"""
        self.stop()
        self.hide()
        if SI.loginWin:
            SI.loginWin.show()
        else:
            SI.loginWin = Win_Login()
            SI.loginWin.show()

    def closeEvent(self, event):
        """重写关闭事件"""
        self.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    SI.loginWin = Win_Login()
    SI.loginWin.show()
    sys.exit(app.exec())