from PyQt6 import QtWidgets, QtCore, QtGui
import cv2, os, time
from threading import Thread
# 关闭YOLO调试信息
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO


class MWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        # 设置界面
        self.setupUI()

        # 绑定按钮事件
        self.camBtn.clicked.connect(self.startCamera)
        self.stopBtn.clicked.connect(self.stop)
        self.videoBtn.clicked.connect(self.openVideoFile)  # 新增视频文件按钮事件

        # 定义定时器，控制视频显示帧率
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        # 加载YOLO模型
        self.model = YOLO('yolov8n.pt')

        # 视频帧处理队列
        self.frameToAnalyze = []

        # 启动帧分析线程
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

    def setupUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('YOLO-Qt 演示（PyQt6版）')

        # 中心部件
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        # 主布局
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 上半部分：视频显示区域
        topLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_treated = QtWidgets.QLabel(self)
        # 设置最小尺寸和边框样式
        self.label_ori_video.setMinimumSize(520, 400)
        self.label_treated.setMinimumSize(520, 400)
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')
        # 添加到布局
        topLayout.addWidget(self.label_ori_video)
        topLayout.addWidget(self.label_treated)
        mainLayout.addLayout(topLayout)

        # 下半部分：日志和按钮
        groupBox = QtWidgets.QGroupBox(self)
        bottomLayout = QtWidgets.QHBoxLayout(groupBox)

        # 日志显示框
        self.textLog = QtWidgets.QTextBrowser()
        bottomLayout.addWidget(self.textLog)

        # 按钮布局
        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn = QtWidgets.QPushButton('📹摄像头')
        self.stopBtn = QtWidgets.QPushButton('🛑停止')
        # 设置按钮大小
        self.videoBtn.setMinimumHeight(40)
        self.camBtn.setMinimumHeight(40)
        self.stopBtn.setMinimumHeight(40)
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.stopBtn)
        bottomLayout.addLayout(btnLayout)

        mainLayout.addWidget(groupBox)

    def startCamera(self):
        """启动摄像头"""
        # 释放可能存在的旧资源
        if hasattr(self, 'cap'):
            self.cap.release()

        # 打开摄像头（Windows使用CAP_DSHOW加速）
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.textLog.append("❌ 摄像头打开失败")
            return

        self.textLog.append("✅ 摄像头启动成功")
        if not self.timer_camera.isActive():
            self.timer_camera.start(50)  # 20fps（1000/50=20）

    def openVideoFile(self):
        """打开本地视频文件"""
        # 释放可能存在的旧资源
        if hasattr(self, 'cap'):
            self.cap.release()

        # 打开文件选择对话框
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if not filePath:
            return

        # 打开视频文件
        self.cap = cv2.VideoCapture(filePath)
        if not self.cap.isOpened():
            self.textLog.append(f"❌ 视频文件打开失败：{filePath}")
            return

        self.textLog.append(f"✅ 视频文件打开成功：{os.path.basename(filePath)}")
        if not self.timer_camera.isActive():
            self.timer_camera.start(30)  # 视频播放帧率（约33fps）

    def show_camera(self):
        """读取视频帧并显示原始画面"""
        if not hasattr(self, 'cap'):
            return

        ret, frame = self.cap.read()
        if not ret:
            self.textLog.append("⚠️ 视频流已结束")
            self.stop()
            return

        # 调整帧大小并转换色彩空间
        frame = cv2.resize(frame, (520, 400))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV默认BGR，转换为RGB

        # 转换为QImage并显示
        q_img = QtGui.QImage(
            frame_rgb.data,
            frame_rgb.shape[1],
            frame_rgb.shape[0],
            QtGui.QImage.Format.Format_RGB888
        )
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(q_img))

        # 若队列空闲，添加帧到分析队列
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame_rgb)

    def frameAnalyzeThreadFunc(self):
        """帧分析线程：处理YOLO目标检测"""
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue

            # 取出帧进行分析
            frame = self.frameToAnalyze.pop(0)
            results = self.model(frame)[0]  # YOLO检测

            # 绘制检测结果
            img_with_boxes = results.plot(line_width=1)

            # 转换为QImage并显示
            q_img = QtGui.QImage(
                img_with_boxes.data,
                img_with_boxes.shape[1],
                img_with_boxes.shape[0],
                QtGui.QImage.Format.Format_RGB888
            )
            self.label_treated.setPixmap(QtGui.QPixmap.fromImage(q_img))

            # 输出检测日志
            detected_classes = [results.names[int(cls)] for cls in results.boxes.cls]
            if detected_classes:
                self.textLog.append(f"🔍 检测到：{', '.join(set(detected_classes))}")

    def stop(self):
        """停止视频流和分析"""
        if self.timer_camera.isActive():
            self.timer_camera.stop()

        if hasattr(self, 'cap'):
            self.cap.release()
            delattr(self, 'cap')  # 移除cap属性

        self.label_ori_video.clear()
        self.label_treated.clear()
        self.textLog.append("🛑 已停止视频处理")


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MWindow()
    window.show()
    sys.exit(app.exec())
