from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox, QMainWindow, QFileDialog
from PyQt6.uic import loadUi
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import sys
import cv2
import os
import time
import pymysql
from threading import Thread
import torch
import warnings
from pathlib import Path

# 关闭YOLO调试信息
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO


# 共享实例类（管理登录/主窗口实例）
class SI:
    loginWin = None
    mainWin = None


# 1. 登录验证线程（保留原有功能，确保数据库交互稳定）
class LoginThread(QThread):
    result_signal = pyqtSignal(bool, str)

    def __init__(self, username, password):
        super().__init__()
        self.username = username
        self.password = password

        # 数据库连接配置（根据实际环境调整）
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
            # 数据库连接（增加超时控制）
            conn = pymysql.connect(**self.db_config, connect_timeout=10)
            cursor = conn.cursor(pymysql.cursors.DictCursor)

            # 登录查询SQL（避免SQL注入）
            sql = "SELECT * FROM user WHERE account = %s AND password = %s"
            cursor.execute(sql, (self.username, self.password))
            result = cursor.fetchone()

            # 验证结果返回
            if result:
                self.result_signal.emit(True, "登录成功")
            else:
                self.result_signal.emit(False, "账号或密码错误")

            # 关闭资源
            cursor.close()
            conn.close()

        except pymysql.MySQLError as e:
            error_msg = f"数据库错误：{str(e)}"
            print(f"子线程：{error_msg}")
            self.result_signal.emit(False, error_msg)
        except Exception as e:
            error_msg = f"未知错误：{str(e)}"
            print(f"子线程：{error_msg}")
            self.result_signal.emit(False, error_msg)


# 2. 登录窗口类（保留原有UI逻辑，确保控件绑定正常）
class Win_Login(QWidget):
    def __init__(self):
        super().__init__()
        try:
            # 加载登录UI文件
            self.ui = loadUi("D:/qtdesigner/jiance.ui", self)
            print("登录窗口：UI文件加载成功")
            self.init_login_ui()
        except Exception as e:
            error_msg = f"UI加载失败：{str(e)}"
            print(f"登录窗口：{error_msg}")
            QMessageBox.critical(self, "错误", error_msg)
            return

    def init_login_ui(self):
        """初始化登录窗口控件绑定"""
        try:
            # 绑定登录按钮和回车键事件
            self.ui.sign_in.clicked.connect(self.on_sign_in)
            self.ui.enter_the_password.returnPressed.connect(self.on_sign_in)
            # 设置密码框回声模式
            self.ui.enter_the_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
            print("登录窗口：事件绑定成功")
        except AttributeError as e:
            error_msg = f"控件不存在：{str(e)}（请检查UI文件中控件名称）"
            print(f"登录窗口：{error_msg}")
            QMessageBox.critical(self, "错误", error_msg)

    def on_sign_in(self):
        """处理登录逻辑"""
        print("\n登录按钮被点击，开始处理...")
        try:
            # 获取用户输入（去空格）
            username = self.ui.lineEdit.text().strip()
            password = self.ui.enter_the_password.text().strip()
            print(f"输入的用户名: {username}, 密码: ******")

            # 空值校验
            if not username or not password:
                QMessageBox.warning(self, "输入错误", "用户名和密码不能为空！")
                return

            # 启动登录线程（避免阻塞UI）
            self.login_thread = LoginThread(username, password)
            self.login_thread.result_signal.connect(self.handle_login_result)
            self.login_thread.start()
            print("登录子线程已启动，等待结果...")

        except Exception as e:
            error_msg = f"登录处理出错：{str(e)}"
            print(f"on_sign_in方法：{error_msg}")
            QMessageBox.critical(self, "错误", error_msg)

    def handle_login_result(self, success, msg):
        """处理登录结果回调"""
        print(f"\n收到登录结果：成功={success}, 消息={msg}")
        if success:
            QMessageBox.information(self, "成功", msg)
            # 打开主窗口并隐藏登录窗口
            try:
                SI.mainWin = Win_Main()
                SI.mainWin.show()
                self.ui.enter_the_password.setText('')  # 清空密码
                self.hide()
            except Exception as e:
                error_msg = f"打开主窗口失败：{str(e)}"
                print(f"handle_login_result：{error_msg}")
                QMessageBox.critical(self, "错误", error_msg)
        else:
            QMessageBox.warning(self, "失败", msg)


# 3. 主窗口类（核心功能：视频播放实时YOLOv8检测）
class Win_Main(QWidget):
    # 信号定义（主线程更新UI）
    update_log_signal = pyqtSignal(str)  # 日志更新

    def __init__(self):
        super().__init__()
        # 信号绑定（日志更新）
        self.update_log_signal.connect(self.append_log)

        # 状态标记
        self.is_stopping = False  # 停止状态标记（避免重复停止）
        self.detected_plastic_bag = False  # 塑料袋检测状态
        self.use_designer_ui = True  # 是否使用设计器UI
        self.ui = None  # UI实例
        self.cap = None  # 视频/摄像头捕获对象
        self.timer_camera = None  # 摄像头/视频定时器
        self.model = None  # YOLOv8全类别检测模型
        self.yolo_plastic_model = None  # 塑料袋专项检测模型

        # 初始化UI（优先加载设计器UI，失败则代码创建）
        self.init_ui()
        # 初始化核心组件（模型、资源）
        self.init_core_components()
        # 绑定事件（按钮、复选框等）
        self.bind_all_events()

    def init_ui(self):
        """初始化主窗口UI"""
        try:
            # 尝试加载设计器UI
            self.ui = loadUi("D:/qtdesigner/main11111.ui", self)
            print("主窗口：设计器UI加载成功")
            self.setup_designer_ui()
        except Exception as e:
            error_msg = f"设计器UI加载失败：{str(e)}"
            print(f"主窗口：{error_msg}")
            QMessageBox.warning(self, "警告", f"{error_msg}，将使用代码创建基础UI")
            self.use_designer_ui = False
            self.setup_code_ui()

    def setup_designer_ui(self):
        """从设计器UI中获取控件并初始化"""
        try:
            # 视频显示标签（原始+检测后）
            self.label_ori_video = self.ui.label_ori_video
            self.label_treated = self.ui.label_treated
            # 日志文本框
            self.textLog = self.ui.textLog
            # 功能按钮
            self.videoBtn = self.ui.videoBtn
            self.camBtn = self.ui.camBtn
            self.stopBtn = self.ui.stopBtn
            # 塑料袋检测复选框
            self.detect_plastic_bag_intrusion = getattr(self.ui, "detect_plastic_bag_intrusion", None)

            # 控件有效性校验
            if not self.detect_plastic_bag_intrusion:
                QMessageBox.warning(self, "控件警告", "未找到'塑料袋入侵检测'复选框，将自动添加")
                self.add_plastic_checkbox()

            # 设置控件样式和尺寸
            self.label_ori_video.setMinimumSize(520, 400)
            self.label_treated.setMinimumSize(520, 400)
            self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
            self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')
            self.textLog.setReadOnly(True)

        except AttributeError as e:
            error_msg = f"UI控件初始化失败：{str(e)}"
            print(f"setup_designer_ui：{error_msg}")
            QMessageBox.warning(self, "警告", error_msg)

    def setup_code_ui(self):
        """代码创建基础UI（设计器UI加载失败时备用）"""
        self.setWindowTitle("YOLOv8检测系统（基础UI）")
        self.resize(1300, 700)

        # 主布局
        main_layout = QtWidgets.QVBoxLayout(self)
        # 视频显示布局（左右分栏）
        video_layout = QtWidgets.QHBoxLayout()
        # 按钮布局
        btn_layout = QtWidgets.QHBoxLayout()
        # 日志布局
        log_layout = QtWidgets.QVBoxLayout()

        # 1. 视频显示标签
        self.label_ori_video = QtWidgets.QLabel("原始视频")
        self.label_ori_video.setMinimumSize(520, 400)
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_ori_video.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_treated = QtWidgets.QLabel("YOLOv8检测后视频")
        self.label_treated.setMinimumSize(520, 400)
        self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_treated.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 2. 功能按钮
        self.videoBtn = QtWidgets.QPushButton("打开视频（YOLOv8检测）")
        self.camBtn = QtWidgets.QPushButton("启动摄像头（YOLOv8检测）")
        self.stopBtn = QtWidgets.QPushButton("停止")
        self.detect_plastic_bag_intrusion = QtWidgets.QCheckBox("塑料袋专项检测")

        # 3. 日志文本框
        self.textLog = QtWidgets.QTextEdit()
        self.textLog.setReadOnly(True)
        self.textLog.setPlaceholderText("检测日志（YOLOv8结果将显示在这里）...")
        self.textLog.setMinimumHeight(150)

        # 组装布局
        video_layout.addWidget(self.label_ori_video)
        video_layout.addSpacing(20)
        video_layout.addWidget(self.label_treated)

        btn_layout.addWidget(self.videoBtn)
        btn_layout.addWidget(self.camBtn)
        btn_layout.addWidget(self.stopBtn)
        btn_layout.addSpacing(30)
        btn_layout.addWidget(self.detect_plastic_bag_intrusion)
        btn_layout.addStretch()

        log_layout.addWidget(self.textLog)

        main_layout.addLayout(video_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(log_layout)

    def add_plastic_checkbox(self):
        """为设计器UI补充塑料袋检测复选框（当控件缺失时）"""
        try:
            # 查找按钮布局（假设按钮在水平布局中）
            btn_layout = self.videoBtn.parent().layout()
            if isinstance(btn_layout, QtWidgets.QHBoxLayout):
                self.detect_plastic_bag_intrusion = QtWidgets.QCheckBox("塑料袋专项检测")
                btn_layout.addWidget(self.detect_plastic_bag_intrusion)
                print("主窗口：已补充塑料袋检测复选框")
            else:
                print("主窗口：无法找到按钮布局，无法补充复选框")
        except Exception as e:
            print(f"add_plastic_checkbox：{str(e)}")

    def init_core_components(self):
        """初始化核心组件（YOLOv8模型、定时器等）"""
        # 1. 初始化定时器（控制帧读取速度）
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera_with_yolo)  # 绑定：帧显示+YOLO检测

        # 2. 加载YOLOv8全类别模型（视频/摄像头默认检测模型）
        try:
            model_path = Path("yolov8n.pt")
            if not model_path.exists():
                self.append_log("ℹ️ 本地未找到yolov8n.pt，将自动下载（约6MB，需网络通畅）...")
            self.model = YOLO(model_path)  # 加载YOLOv8轻量级模型

            # 模型有效性测试
            test_frame = torch.zeros((3, 640, 640))  # 3通道640x640测试帧
            test_result = self.model(test_frame, verbose=False)
            if isinstance(test_result, list) and len(test_result) > 0 and hasattr(test_result[0], 'boxes'):
                self.append_log("✅ YOLOv8模型加载成功，支持80类目标检测（如人、车、动物等）")
            else:
                raise Exception("模型推理结果异常，无有效检测框（boxes）")
        except Exception as e:
            error_msg = f"YOLOv8模型加载失败：{str(e)}"
            self.append_log(f"❌ {error_msg}")
            QMessageBox.critical(self, "模型错误", f"{error_msg}\n请检查网络（首次加载需下载模型）或模型文件完整性")
            self.model = None

        # 3. 加载塑料袋专项检测模型（复用YOLOv8，仅检测指定类别）
        try:
            self.yolo_plastic_model = YOLO("yolov8n.pt")  # 复用模型文件，减少内存占用
            self.append_log("✅ 塑料袋专项模型加载成功，检测类别：24=背包、26=手提包、41=购物袋")
        except Exception as e:
            error_msg = f"塑料袋专项模型加载失败：{str(e)}"
            self.append_log(f"❌ {error_msg}")
            QMessageBox.warning(self, "模型警告", error_msg)
            self.yolo_plastic_model = None

    def bind_all_events(self):
        """绑定所有控件事件"""
        # 视频按钮：模型加载成功才启用
        if hasattr(self, 'videoBtn'):
            if self.model:
                self.videoBtn.clicked.connect(self.open_video_file)
                self.videoBtn.setEnabled(True)
            else:
                self.videoBtn.setEnabled(False)
                self.append_log("⚠️ 视频按钮已禁用（YOLOv8模型未加载）")

        # 摄像头按钮：模型加载成功才启用
        if hasattr(self, 'camBtn'):
            if self.model:
                self.camBtn.clicked.connect(self.start_camera)
                self.camBtn.setEnabled(True)
            else:
                self.camBtn.setEnabled(False)
                self.append_log("⚠️ 摄像头按钮已禁用（YOLOv8模型未加载）")

        # 停止按钮：始终启用
        if hasattr(self, 'stopBtn'):
            self.stopBtn.clicked.connect(self.stop_all)

        # 塑料袋专项检测复选框：模型加载成功才绑定
        if self.detect_plastic_bag_intrusion and self.yolo_plastic_model:
            self.detect_plastic_bag_intrusion.stateChanged.connect(self.handle_plastic_intrusion)
        elif self.detect_plastic_bag_intrusion:
            self.detect_plastic_bag_intrusion.setEnabled(False)
            self.append_log("⚠️ 塑料袋检测复选框已禁用（专项模型未加载）")

    # -------------------------- 核心功能：视频文件读取+YOLOv8实时检测 --------------------------
    def open_video_file(self):
        """打开视频文件，启用YOLOv8实时检测"""
        # 1. 先停止现有任务（避免资源冲突）
        self.stop_all()

        # 2. 视频文件选择对话框（支持多种格式）
        video_formats = (
            "所有视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.rmvb "
            "*.mpg *.mpeg *.3gp *.webm *.ts *.m4v *.f4v *.asf);;"
            "MP4格式 (*.mp4);;AVI格式 (*.avi);;MOV格式 (*.mov);;MKV格式 (*.mkv)"
        )
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="选择视频文件（YOLOv8实时检测）",
            directory="",
            filter=video_formats
        )
        if not file_path:
            self.append_log("ℹ️ 用户取消视频选择")
            return

        # 3. 路径有效性校验（处理中文/空格路径）
        if not os.path.exists(file_path):
            error_msg = f"所选文件不存在：{os.path.basename(file_path)}"
            self.append_log(f"❌ {error_msg}")
            QMessageBox.warning(self, "文件错误", error_msg)
            return
        try:
            file_path = file_path.encode('gbk').decode('gbk')  # 兼容Windows中文路径
        except:
            pass

        # 4. 打开视频（尝试FFmpeg解码器，兼容特殊编码）
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                error_msg = f"无法打开视频：{os.path.basename(file_path)}（编码不支持或文件损坏）"
                self.append_log(f"❌ {error_msg}")
                QMessageBox.warning(self, "视频错误", error_msg)
                return

        # 5. 按视频实际帧率设置定时器（避免帧丢失/过快）
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 25  # 默认帧率（防止异常值）
        timer_interval = int(1000 / video_fps)  # 定时器间隔（毫秒）
        self.timer_camera.start(timer_interval)

        # 6. 日志提示
        self.append_log(f"✅ 视频打开成功：{os.path.basename(file_path)}（帧率：{video_fps:.1f}，已启用YOLOv8检测）")

    # -------------------------- 核心功能：摄像头启动+YOLOv8实时检测 --------------------------
    def start_camera(self):
        """启动摄像头，启用YOLOv8实时检测"""
        # 1. 先停止现有任务
        self.stop_all()
        self.append_log("ℹ️ 尝试启动摄像头（YOLOv8检测模式）...")

        # 2. 释放旧摄像头资源
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.append_log("ℹ️ 已释放旧摄像头资源")

        # 3. 尝试打开摄像头（支持多设备，优先设备0）
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)  # 重试外接摄像头
            if not self.cap.isOpened():
                error_msg = "摄像头无法打开（设备被占用、不存在或权限不足）"
                self.append_log(f"❌ {error_msg}")
                QMessageBox.warning(self, "摄像头错误", error_msg)
                return

        # 4. 验证摄像头帧读取能力
        ret, test_frame = self.cap.read()
        if not ret:
            error_msg = "摄像头无法读取帧（设备故障或格式不支持）"
            self.append_log(f"❌ {error_msg}")
            self.cap.release()
            self.cap = None
            QMessageBox.warning(self, "摄像头错误", error_msg)
            return

        # 5. 启动定时器（摄像头默认20fps，间隔50ms）
        self.timer_camera.start(50)
        self.append_log("✅ 摄像头启动成功，已启用YOLOv8实时检测")

    # -------------------------- 核心功能：帧显示+YOLOv8实时检测 --------------------------
    def show_camera_with_yolo(self):
        """读取帧→显示原始视频→YOLOv8检测→显示检测后视频"""
        try:
            if self.is_stopping or self.cap is None or self.model is None:
                return

            # 帧读取重试（最多3次，避免偶尔丢帧）
            ret = False
            frame = None
            for _ in range(3):
                ret, frame = self.cap.read()
                if ret:
                    break

            if not ret:
                # 判断是否视频结束（摄像头卡顿不会触发此逻辑）
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    self.append_log("ℹ️ 视频播放结束，自动停止检测")
                    self.stop_all()  # 视频结束后自动停止并清屏
                else:
                    self.append_log("⚠️ 暂时无法读取帧（设备卡顿，将重试）")
                return

            # 1. 显示原始视频
            self.show_original_video(frame)

            # 2. YOLOv8实时检测（置信度阈值0.25，过滤低置信度结果）
            detected_frame, detected_classes = self.yolo_detect_frame(frame)

            # 3. 显示检测后视频（带检测框和类别标签）
            self.show_detected_video(detected_frame)

            # 4. 日志记录检测结果（去重，避免重复打印）
            if detected_classes:
                unique_classes = list(set(detected_classes))
                self.update_log_signal.emit(f"🔍 检测到：{', '.join(unique_classes)}")

        except Exception as e:
            if not self.is_stopping:
                error_msg = f"帧处理/检测出错：{str(e)}"
                self.update_log_signal.emit(f"❌ {error_msg}")
                print(f"show_camera_with_yolo：{error_msg}")

    def yolo_detect_frame(self, frame):
        """单帧YOLOv8检测（返回带检测框的帧和检测到的类别列表）"""
        detected_frame = frame.copy()  # 默认返回原始帧（避免检测失败黑屏）
        detected_classes = []
        try:
            # YOLOv8检测（全类别，置信度阈值0.25）
            results = self.model(frame, conf=0.25, verbose=False)

            # 结果格式校验（避免异常崩溃）
            if isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'boxes'):
                # 绘制检测框（line_width=2：框线宽度，font_size=10：标签字体大小）
                detected_frame = results[0].plot(line_width=2, font_size=10)

                # 提取检测到的类别名称
                if hasattr(results[0].boxes, 'cls') and len(results[0].boxes.cls) > 0:
                    detected_classes = [self.model.names[int(cls.item())] for cls in results[0].boxes.cls]
            else:
                self.update_log_signal.emit("⚠️ YOLOv8检测结果格式异常，跳过当前帧")
        except Exception as e:
            self.update_log_signal.emit(f"❌ YOLOv8单帧检测出错：{str(e)}")
        return detected_frame, detected_classes

    # -------------------------- 核心功能：塑料袋专项检测 --------------------------
    # def handle_plastic_intrusion(self, state):
    #     """塑料袋检测复选框逻辑：选中时启动专项检测"""
    #     if state == Qt.CheckState.Checked.value and self.yolo_plastic_model:
    #         self.stop_all()  # 停止现有任务
    #         self.open_plastic_detect_video()  # 启动专项检测
    #         self.detect_plastic_bag_intrusion.setChecked(False)  # 检测后取消选中
    #
    # def open_plastic_detect_video(self):
    #     """塑料袋专项检测（仅检测背包/手提包/购物袋）"""
    #     if not self.yolo_plastic_model:
    #         QMessageBox.warning(self, "模型错误", "塑料袋专项模型未加载，无法执行检测")
    #         return
    #
    #     # 视频选择对话框
    #     video_formats = (
    #         "所有视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.rmvb "
    #         "*.mpg *.mpeg *.3gp *.webm *.ts *.m4v *.f4v *.asf);;"
    #         "MP4格式 (*.mp4);;AVI格式 (*.avi);;MOV格式 (*.mov);;MKV格式 (*.mkv)"
    #     )
    #     file_path, _ = QFileDialog.getOpenFileName(
    #         parent=self,
    #         caption="选择视频文件（塑料袋专项检测）",
    #         directory="",
    #         filter=video_formats
    #     )
    #     if not file_path:
    #         self.append_log("ℹ️ 用户取消塑料袋检测视频选择")
    #         return
    #
    #     # 路径校验
    #     if not os.path.exists(file_path):
    #         error_msg = f"所选文件不存在：{os.path.basename(file_path)}"
    #         self.append_log(f"❌ {error_msg}")
    #         QMessageBox.warning(self, "文件错误", error_msg)
    #         return
    #     try:
    #         file_path = file_path.encode('gbk').decode('gbk')
    #     except:
    #         pass
    #
    #     # 初始化检测状态
    #     self.detected_plastic_bag = False
    #     max_confidence = 0.0
    #     self.label_ori_video.clear()
    #     self.label_treated.clear()
    #     self.append_log(f"ℹ️ 开始塑料袋专项检测：{os.path.basename(file_path)}（仅检测背包/手提包/购物袋）")
    #
    #     # 打开视频
    #     self.cap = cv2.VideoCapture(file_path)
    #     if not self.cap.isOpened():
    #         self.cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
    #         if not self.cap.isOpened():
    #             error_msg = f"无法打开视频：{os.path.basename(file_path)}"
    #             self.append_log(f"❌ {error_msg}")
    #             QMessageBox.warning(self, "视频错误", error_msg)
    #             return
    #
    #     # 逐帧专项检测（独立循环，不依赖定时器）
    #     while self.cap.isOpened() and not self.is_stopping:
    #         ret, frame = self.cap.read()
    #         if not ret:
    #             break  # 视频结束
    #
    #         # 1. 显示原始视频
    #         self.show_original_video(frame)
    #         # 2. 塑料袋专项检测（仅检测24/26/41类，置信度阈值0.3）
    #         detected_frame, current_conf = self.detect_plastic_bag_frame(frame)
    #         if current_conf > max_confidence:
    #             max_confidence = current_conf
    #         # 3. 显示检测后视频
    #         self.show_detected_video(detected_frame)
    #         # 4. 处理UI事件（避免卡顿）
    #         QApplication.processEvents()
    #         # 5. 控制播放速度（模拟25fps）
    #         time.sleep(0.04)
    #
    #     # 检测结束处理
    #     self.cap.release()
    #     self.cap = None
    #     self.append_log("ℹ️ 塑料袋专项检测结束")
    #     # 结果提示
    #     if self.detected_plastic_bag:
    #         QMessageBox.warning(
    #             self, "检测警告",
    #             f"在视频《{os.path.basename(file_path)}》中检测到塑料袋相关物品！\n最高置信度：{max_confidence:.2f}"
    #         )
    #         self.append_log(f"⚠️ 专项检测结果：发现塑料袋相关物品（最高置信度：{max_confidence:.2f}）")
    #     else:
    #         QMessageBox.information(
    #             self, "检测结果",
    #             f"在视频《{os.path.basename(file_path)}》中未检测到塑料袋相关物品。\n最高置信度：{max_confidence:.2f}"
    #         )
    #         self.append_log(f"ℹ️ 专项检测结果：未发现塑料袋相关物品")
    #
    # def detect_plastic_bag_frame(self, frame):
    #     """单帧塑料袋专项检测"""
    #     max_conf = 0.0
    #     detected_frame = frame.copy()
    #     try:
    #         # YOLOv8检测（仅24=背包、26=手提包、41=购物袋，置信度阈值0.3）
    #         results = self.yolo_plastic_model(frame, classes=[24, 26, 41], conf=0.3, verbose=False)
    #
    #         # 结果格式校验
    #         if isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'boxes'):
    #             detected_frame = results[0].plot(line_width=2, font_size=10)
    #             # 更新检测状态和最高置信度
    #             if len(results[0].boxes) > 0:
    #                 self.detected_plastic_bag = True
    #                 if hasattr(results[0].boxes, 'conf') and len(results[0].boxes.conf) > 0:
    #                     max_conf = max(box.item() for box in results[0].boxes.conf)
    #         else:
    #             self.append_log("⚠️ 塑料袋专项检测结果格式异常，跳过当前帧")
    #     except Exception as e:
    #         self.append_log(f"❌ 塑料袋单帧检测出错：{str(e)}")
    #     return detected_frame, max_conf
    #
    # # -------------------------- 视频显示辅助函数 --------------------------
    # def show_original_video(self, frame):
    #     """显示原始视频到label_ori_video（布满标签区域）"""
    #     ori_size = self.label_ori_video.size()
    #     target_w, target_h = ori_size.width(), ori_size.height()
    #     if target_w <= 0 or target_h <= 0:
    #         target_w, target_h = 520, 400  # 默认尺寸
    #
    #     # 缩放帧（保持宽高比，填满标签）
    #     frame_h, frame_w = frame.shape[:2]
    #     scale = max(target_w / frame_w, target_h / frame_h)
    #     new_w, new_h = int(frame_w * scale), int(frame_h * scale)
    #     frame_resized = cv2.resize(frame, (new_w, new_h))
    #
    #     # BGR转RGB（Qt显示格式）
    #     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    #     # 转换为QImage并显示
    #     h, w, ch = frame_rgb.shape
    #     q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    #     self.label_ori_video.setPixmap(QPixmap.fromImage(q_img).scaled(
    #         ori_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    #     ))
    #
    # def show_detected_video(self, frame):
    #     """显示检测后视频到label_treated（布满标签区域）"""
    #     treated_size = self.label_treated.size()
    #     target_w, target_h = treated_size.width(), treated_size.height()
    #     if target_w <= 0 or target_h <= 0:
    #         target_w, target_h = 520, 400  # 默认尺寸
    #
    #     # 缩放帧（保持宽高比，填满标签）
    #     frame_h, frame_w = frame.shape[:2]
    #     scale = max(target_w / frame_w, target_h / frame_h)
    #     new_w, new_h = int(frame_w * scale), int(frame_h * scale)
    #     frame_resized = cv2.resize(frame, (new_w, new_h))
    #
    #     # BGR转RGB（Qt显示格式）
    #     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    #     # 转换为QImage并显示
    #     h, w, ch = frame_rgb.shape
    #     q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    #     self.label_treated.setPixmap(QPixmap.fromImage(q_img).scaled(
    #         treated_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    #     ))
    def handle_plastic_intrusion(self, state):
        """塑料袋检测复选框逻辑：选中时启动专项检测（支持视频和照片）"""
        if state == Qt.CheckState.Checked.value and self.yolo_plastic_model:
            self.stop_all()  # 停止现有任务
            self.open_plastic_detect_file()  # 启动专项检测（修改方法名）
            self.detect_plastic_bag_intrusion.setChecked(False)  # 检测后取消选中

    def open_plastic_detect_file(self):
        """塑料袋专项检测（支持视频和照片）"""
        if not self.yolo_plastic_model:
            QMessageBox.warning(self, "模型错误", "塑料袋专项模型未加载，无法执行检测")
            return

        # 支持视频和照片的文件选择对话框
        file_formats = (
            "所有支持的文件 (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.gif);;"
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;"
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif);;"
        )
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="选择文件（塑料袋专项检测）",
            directory="",
            filter=file_formats
        )
        if not file_path:
            self.append_log("ℹ️ 用户取消塑料袋检测文件选择")
            return

        # 路径校验
        if not os.path.exists(file_path):
            error_msg = f"所选文件不存在：{os.path.basename(file_path)}"
            self.append_log(f"❌ {error_msg}")
            QMessageBox.warning(self, "文件错误", error_msg)
            return
        try:
            file_path = file_path.encode('gbk').decode('gbk')
        except:
            pass

        # 根据文件扩展名判断是视频还是照片
        file_ext = os.path.splitext(file_path)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        if file_ext in image_extensions:
            # 处理照片检测
            self.detect_plastic_in_photo(file_path)
        else:
            # 处理视频检测
            self.detect_plastic_in_video(file_path)

    def detect_plastic_in_video(self, file_path):
        """在视频中检测塑料袋相关物品"""
        # 初始化检测状态
        self.detected_plastic_bag = False
        max_confidence = 0.0
        self.label_ori_video.clear()
        self.label_treated.clear()
        self.append_log(f"ℹ️ 开始视频塑料袋专项检测：{os.path.basename(file_path)}（仅检测背包/手提包/购物袋）")

        # 打开视频
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                error_msg = f"无法打开视频：{os.path.basename(file_path)}"
                self.append_log(f"❌ {error_msg}")
                QMessageBox.warning(self, "视频错误", error_msg)
                return

        # 逐帧专项检测
        while self.cap.isOpened() and not self.is_stopping:
            ret, frame = self.cap.read()
            if not ret:
                break  # 视频结束

            # 1. 显示原始视频
            self.show_original_video(frame)
            # 2. 塑料袋专项检测
            detected_frame, current_conf = self.detect_plastic_bag_frame(frame)
            if current_conf > max_confidence:
                max_confidence = current_conf
            # 3. 显示检测后视频
            self.show_detected_video(detected_frame)
            # 4. 处理UI事件（避免卡顿）
            QApplication.processEvents()
            # 5. 控制播放速度（模拟25fps）
            time.sleep(0.04)

        # 检测结束处理
        self.cap.release()
        self.cap = None
        self.append_log("ℹ️ 视频塑料袋专项检测结束")
        # 结果提示
        self.show_plastic_detection_result(file_path, max_confidence)

    def detect_plastic_in_photo(self, file_path):
        """在照片中检测塑料袋相关物品"""
        # 初始化检测状态
        self.detected_plastic_bag = False
        max_confidence = 0.0
        self.label_ori_video.clear()
        self.label_treated.clear()
        self.append_log(f"ℹ️ 开始照片塑料袋专项检测：{os.path.basename(file_path)}（仅检测背包/手提包/购物袋）")

        # 读取照片
        frame = cv2.imread(file_path)
        if frame is None:
            error_msg = f"无法读取照片：{os.path.basename(file_path)}（格式不支持或文件损坏）"
            self.append_log(f"❌ {error_msg}")
            QMessageBox.warning(self, "照片错误", error_msg)
            return

        # 1. 显示原始照片
        self.show_original_video(frame)
        # 2. 塑料袋专项检测
        detected_frame, max_confidence = self.detect_plastic_bag_frame(frame)
        # 3. 显示检测后照片
        self.show_detected_video(detected_frame)

        self.append_log("ℹ️ 照片塑料袋专项检测结束")
        # 结果提示
        self.show_plastic_detection_result(file_path, max_confidence)

    def show_plastic_detection_result(self, file_path, max_confidence):
        """显示塑料袋检测结果（视频和照片通用）"""
        if self.detected_plastic_bag:
            QMessageBox.warning(
                self, "检测警告",
                f"在《{os.path.basename(file_path)}》中检测到塑料袋相关物品！\n最高置信度：{max_confidence:.2f}"
            )
            self.append_log(f"⚠️ 专项检测结果：发现塑料袋相关物品（最高置信度：{max_confidence:.2f}）")
        else:
            QMessageBox.information(
                self, "检测结果",
                f"在《{os.path.basename(file_path)}》中未检测到塑料袋相关物品。\n最高置信度：{max_confidence:.2f}"
            )
            self.append_log(f"ℹ️ 专项检测结果：未发现塑料袋相关物品")

    def detect_plastic_bag_frame(self, frame):
        """单帧塑料袋专项检测（视频和照片通用）"""
        max_conf = 0.0
        detected_frame = frame.copy()
        try:
            # YOLOv8检测（仅24=背包、26=手提包、41=购物袋，置信度阈值0.3）
            results = self.yolo_plastic_model(frame, classes=[0], conf=0.3, verbose=False)

            # 结果格式校验
            if isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'boxes'):
                detected_frame = results[0].plot(line_width=2, font_size=10)
                # 更新检测状态和最高置信度
                if len(results[0].boxes) > 0:
                    self.detected_plastic_bag = True
                    if hasattr(results[0].boxes, 'conf') and len(results[0].boxes.conf) > 0:
                        max_conf = max(box.item() for box in results[0].boxes.conf)
            else:
                self.append_log("⚠️ 塑料袋专项检测结果格式异常")
        except Exception as e:
            self.append_log(f"❌ 塑料袋单帧检测出错：{str(e)}")
        return detected_frame, max_conf

    # 视频/图片显示辅助函数保持不变
    def show_original_video(self, frame):
        """显示原始视频/照片到label_ori_video（布满标签区域）"""
        ori_size = self.label_ori_video.size()
        target_w, target_h = ori_size.width(), ori_size.height()
        if target_w <= 0 or target_h <= 0:
            target_w, target_h = 520, 400  # 默认尺寸

        # 缩放帧（保持宽高比，填满标签）
        frame_h, frame_w = frame.shape[:2]
        scale = max(target_w / frame_w, target_h / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))

        # BGR转RGB（Qt显示格式）
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # 转换为QImage并显示
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_ori_video.setPixmap(QPixmap.fromImage(q_img).scaled(
            ori_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def show_detected_video(self, frame):
        """显示检测后视频/照片到label_treated（布满标签区域）"""
        treated_size = self.label_treated.size()
        target_w, target_h = treated_size.width(), treated_size.height()
        if target_w <= 0 or target_h <= 0:
            target_w, target_h = 520, 400  # 默认尺寸

        # 缩放帧（保持宽高比，填满标签）
        frame_h, frame_w = frame.shape[:2]
        scale = max(target_w / frame_w, target_h / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))

        # BGR转RGB（Qt显示格式）
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # 转换为QImage并显示
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_treated.setPixmap(QPixmap.fromImage(q_img).scaled(
            treated_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    # -------------------------- 停止功能（彻底释放资源+清屏） --------------------------
    def stop_all(self):
        """停止所有任务（视频/摄像头/检测），并清空显示标签"""
        if self.is_stopping:
            return

        self.is_stopping = True
        self.append_log("⏳ 正在停止所有任务...")

        # 1. 停止定时器
        if self.timer_camera and self.timer_camera.isActive():
            self.timer_camera.stop()
            self.append_log("ℹ️ 定时器已停止")

        # 2. 释放视频/摄像头资源
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.append_log("ℹ️ 视频/摄像头资源已释放")

        # 3. 清空显示标签（核心：视频结束后清屏）
        self.label_ori_video.clear()
        self.label_treated.clear()

        # 4. 重置停止状态（延迟1秒，确保资源释放完成）
        QtCore.QTimer.singleShot(1000, self.reset_stop_state)

    def reset_stop_state(self):
        """重置停止状态，允许重新启动任务"""
        self.is_stopping = False
        self.append_log("🛑 所有任务已停止，显示标签已清空")

    # -------------------------- 辅助功能 --------------------------
    def append_log(self, text):
        """添加带时间戳的日志，并自动滚动到最新行"""
        current_time = time.strftime("%H:%M:%S", time.localtime())
        self.textLog.append(f"[{current_time}] {text}")
        # 自动滚动到日志底部
        self.textLog.verticalScrollBar().setValue(self.textLog.verticalScrollBar().maximum())

    def logout(self):
        """退出登录，返回登录窗口"""
        self.stop_all()
        self.hide()
        if SI.loginWin:
            SI.loginWin.show()
        else:
            SI.loginWin = Win_Login()
            SI.loginWin.show()

    def closeEvent(self, event):
        """窗口关闭事件（确保资源彻底释放）"""
        self.stop_all()
        event.accept()


# 程序入口（适配高DPI，确保UI显示正常）
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 初始化登录窗口
    SI.loginWin = Win_Login()
    SI.loginWin.show()
    # 启动应用
    sys.exit(app.exec())
