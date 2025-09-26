from PyQt6 import QtWidgets, QtCore, QtGui
import cv2, os, time
from threading import Thread
import numpy as np

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
        self.videoBtn.clicked.connect(self.openVideoFile)
        self.saveBtn.clicked.connect(self.saveResults)  # 新增保存结果按钮

        # 状态变量
        self.is_playing = False
        self.is_paused = False
        self.current_file = None
        self.save_path = None

        # 定义定时器，控制视频显示帧率
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        # 加载YOLO模型
        self.model = YOLO('yolov8n.pt')

        # 视频帧处理队列
        self.frameToAnalyze = []
        self.processedFrames = []  # 用于存储处理后的帧，以便保存

        # 启动帧分析线程
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

    def setupUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('YOLO-Qt 演示（支持视频文件）')

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

        # 视频控制布局（新增）
        videoControlLayout = QtWidgets.QHBoxLayout()
        self.playPauseBtn = QtWidgets.QPushButton('⏸️ 暂停')
        self.playPauseBtn.setMinimumHeight(30)
        self.playPauseBtn.setEnabled(False)
        self.playPauseBtn.clicked.connect(self.togglePlayPause)

        self.speedLabel = QtWidgets.QLabel("播放速度:")
        self.speedSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speedSlider.setMinimum(1)
        self.speedSlider.setMaximum(5)
        self.speedSlider.setValue(2)  # 默认2倍速
        self.speedSlider.valueChanged.connect(self.changePlaybackSpeed)

        videoControlLayout.addWidget(self.playPauseBtn)
        videoControlLayout.addWidget(self.speedLabel)
        videoControlLayout.addWidget(self.speedSlider)
        videoControlLayout.addStretch()
        mainLayout.addLayout(videoControlLayout)

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
        self.saveBtn = QtWidgets.QPushButton('💾保存结果')  # 新增保存按钮
        self.stopBtn = QtWidgets.QPushButton('🛑停止')
        # 设置按钮大小
        self.videoBtn.setMinimumHeight(40)
        self.camBtn.setMinimumHeight(40)
        self.saveBtn.setMinimumHeight(40)
        self.stopBtn.setMinimumHeight(40)
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.saveBtn)
        btnLayout.addWidget(self.stopBtn)
        bottomLayout.addLayout(btnLayout)

        mainLayout.addWidget(groupBox)

    def startCamera(self):
        """启动摄像头"""
        self.stop()  # 先停止任何正在进行的视频或摄像头

        # 打开摄像头（Windows使用CAP_DSHOW加速）
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.textLog.append("❌ 摄像头打开失败")
            return

        self.is_playing = True
        self.is_paused = False
        self.current_file = None
        self.playPauseBtn.setEnabled(False)  # 摄像头模式下暂停按钮不可用
        self.textLog.append("✅ 摄像头启动成功")
        self.timer_camera.start(50)  # 20fps（1000/50=20）

    def openVideoFile(self):
        """打开本地视频文件并播放"""
        self.stop()  # 先停止任何正在进行的视频或摄像头

        # 打开文件选择对话框
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)"
        )
        if not filePath:
            return

        # 打开视频文件
        self.cap = cv2.VideoCapture(filePath)
        if not self.cap.isOpened():
            self.textLog.append(f"❌ 视频文件打开失败：{filePath}")
            return

        # 获取视频信息
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        self.current_file = filePath
        self.is_playing = True
        self.is_paused = False
        self.playPauseBtn.setEnabled(True)
        self.playPauseBtn.setText('⏸️ 暂停')

        # 根据视频帧率设置定时器
        self.original_interval = int(1000 / fps) if fps > 0 else 30
        self.timer_camera.start(self.original_interval)

        self.textLog.append(f"✅ 视频文件打开成功：{os.path.basename(filePath)}")
        self.textLog.append(f"📊 视频信息：{fps:.1f} FPS, {frame_count} 帧, {duration:.1f} 秒")

    def togglePlayPause(self):
        """切换视频播放/暂停状态"""
        if not hasattr(self, 'cap') or self.current_file is None:
            return

        if self.is_paused:
            # 从暂停状态恢复播放
            self.is_paused = False
            self.playPauseBtn.setText('⏸️ 暂停')
            self.timer_camera.start(self.current_interval)
            self.textLog.append("▶️ 继续播放")
        else:
            # 暂停播放
            self.is_paused = True
            self.playPauseBtn.setText('▶️ 播放')
            self.timer_camera.stop()
            self.textLog.append("⏸️ 已暂停")

    def changePlaybackSpeed(self):
        """改变视频播放速度"""
        if not hasattr(self, 'cap') or self.current_file is None or not self.is_playing:
            return

        speed = self.speedSlider.value()
        self.current_interval = max(1, int(self.original_interval / speed))

        if not self.is_paused:
            self.timer_camera.stop()
            self.timer_camera.start(self.current_interval)
            self.textLog.append(f"⚡ 播放速度调整为 {speed}x")

    def show_camera(self):
        """读取视频帧并显示原始画面"""
        if not hasattr(self, 'cap') or self.is_paused:
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
        if len(self.frameToAnalyze) < 5:  # 允许最多5帧的缓冲
            self.frameToAnalyze.append(frame_rgb.copy())

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

            # 存储处理后的帧（用于保存）
            if self.is_playing and self.current_file is not None:
                # 转换回BGR格式用于保存
                bgr_img = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
                self.processedFrames.append(bgr_img)
                # 限制缓存大小，防止内存溢出
                if len(self.processedFrames) > 1000:
                    self.processedFrames = self.processedFrames[-500:]

            # 输出检测日志
            detected_classes = [results.names[int(cls)] for cls in results.boxes.cls]
            if detected_classes:
                unique_classes = list(set(detected_classes))
                self.textLog.append(f"🔍 检测到：{', '.join(unique_classes)}")

    def saveResults(self):
        """保存处理后的视频结果"""
        if not hasattr(self, 'cap') or not self.is_playing or len(self.processedFrames) < 1:
            self.textLog.append("❌ 没有可保存的视频数据，请先播放视频")
            return

        # 获取保存路径
        default_filename = f"processed_{os.path.basename(self.current_file)}"
        filePath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存处理结果", default_filename, "MP4视频 (*.mp4)"
        )
        if not filePath:
            return

        # 确保文件有.mp4扩展名
        if not filePath.endswith('.mp4'):
            filePath += '.mp4'

        # 获取视频信息
        height, width = self.processedFrames[0].shape[:2]
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # 默认帧率

        # 保存视频
        try:
            # 使用MP4编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filePath, fourcc, fps, (width, height))

            # 显示保存进度
            progress_dialog = QtWidgets.QProgressDialog("正在保存视频...", "取消", 0, len(self.processedFrames), self)
            progress_dialog.setWindowTitle("保存中")
            progress_dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

            for i, frame in enumerate(self.processedFrames):
                out.write(frame)
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    out.release()
                    if os.path.exists(filePath):
                        os.remove(filePath)
                    self.textLog.append("❌ 保存已取消")
                    return

            out.release()
            progress_dialog.setValue(len(self.processedFrames))
            self.textLog.append(f"✅ 视频已保存至：{filePath}")
        except Exception as e:
            self.textLog.append(f"❌ 保存失败：{str(e)}")

    def stop(self):
        """停止视频流和分析"""
        if self.timer_camera.isActive():
            self.timer_camera.stop()

        self.is_playing = False
        self.is_paused = False
        self.playPauseBtn.setEnabled(False)
        self.processedFrames.clear()  # 清空处理后的帧缓存

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
