from PyQt6.QtWidgets import (QApplication, QWidget, QFileDialog,
                             QMessageBox, QCheckBox, QLabel)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from PyQt6.uic import loadUi
import sys
import os
import cv2
# 关闭YOLO调试信息
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO

class Stats(QWidget):
    def __init__(self):
        super().__init__()
        # 加载UI文件（请确保路径正确）
        self.ui = loadUi("D:/qtdesigner/main11111.ui", self)

        # 初始化核心组件（关键：获取原始视频标签和检测后视频标签）
        self.init_components()

        # 绑定事件（与设计器控件名匹配）
        self.bind_events()

    def init_components(self):
        """初始化初始化YOLO模型、视频显示控件等核心组件"""
        # 1. 加载YOLOv8模型（检测塑料袋：映射COCO手提包26类、购物袋41类）
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # 首次运行自动下载，后续复用
            print("YOLO模型加载成功")
        except Exception as e:
            QMessageBox.critical(self, "模型错误", f"YOLO模型加载失败：{str(e)}")
            sys.exit(1)

        # 2. 初始化视频捕获对象和检测标记
        self.cap = None
        self.detected_plastic_bag = False

        # -------------------------- 核心修改1：获取两个视频显示控件 --------------------------
        # 获取“原始视频显示标签”（label_ori_video）
        self.label_ori_video = getattr(self.ui, "label_ori_video", None)
        if not self.label_ori_video:
            QMessageBox.warning(self, "控件警告", "未找到原始视频标签'label_ori_video'，请检查UI控件名")

        # 获取“检测后视频显示标签”（label_treated，按需求指定）
        self.label_treated = getattr(self.ui, "label_treated", None)
        if not self.label_treated:
            QMessageBox.warning(self, "控件警告", "未找到检测后视频标签'label_treated'，请检查UI控件名")
        # ----------------------------------------------------------------------------------

    def bind_events(self):
        """绑定UI控件事件"""
        # 1. 视频文件按钮（控件名：videoBtn）
        if hasattr(self.ui, "videoBtn"):
            self.ui.videoBtn.clicked.connect(self.open_and_detect_video)
        else:
            QMessageBox.warning(self, "控件警告", "未找到'视频文件'按钮，请检查UI控件名")

        # 2. 异物入侵复选框（控件名：detect_plastic_bag_intrusion）
        if hasattr(self.ui, "detect_plastic_bag_intrusion"):
            self.ui.detect_plastic_bag_intrusion.stateChanged.connect(self.handle_foreign_invasion)
        else:
            QMessageBox.warning(self, "控件警告", "未找到'异物入侵'复选框，请检查UI控件名")

    def handle_foreign_invasion(self, state):
        """异物入侵复选框选中时，触发视频选择与检测"""
        if state == Qt.CheckState.Checked.value:
            self.open_and_detect_video()
            # 检测完成后取消选中，避免重复触发
            if hasattr(self.ui, "detect_plastic_bag_intrusion"):
                self.ui.detect_plastic_bag_intrusion.setChecked(False)

    # def open_and_detect_video(self):
    #     """打开视频文件，同时显示原始帧（label_ori_video）和检测帧（label_treated）"""
    #     # 支持的视频格式列表
    #     video_formats = "所有视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.rmvb " \
    #                     "*.mpg *.mpeg *.3gp *.webm *.ts *.m4v *.f4v *.asf);;" \
    #                     "MP4格式 (*.mp4);;AVI格式 (*.avi);;MOV格式 (*.mov);;MKV格式 (*.mkv);;FLV格式 (*.flv);;" \
    #                     "所有文件 (*.*)"
    #
    #     # 打开文件选择对话框
    #     file_path, _ = QFileDialog.getOpenFileName(
    #         parent=self,
    #         caption="选择视频文件（支持多格式）",
    #         directory="",
    #         filter=video_formats
    #     )
    #     if not file_path:
    #         return  # 用户取消选择，直接返回
    #
    #     # 检查文件是否存在
    #     if not os.path.exists(file_path):
    #         QMessageBox.warning(self, "文件错误", f"所选文件不存在：{file_path}")
    #         return
    #
    #     # 释放之前的视频资源，清空标签残留画面
    #     if self.cap is not None and self.cap.isOpened():
    #         self.cap.release()
    #         if self.label_ori_video:
    #             self.label_ori_video.clear()
    #         if self.label_treated:
    #             self.label_treated.clear()
    #     self.detected_plastic_bag = False  # 重置检测标记
    #
    #     # 打开选中的视频文件
    #     self.cap = cv2.VideoCapture(file_path)
    #     if not self.cap.isOpened():
    #         QMessageBox.warning(self, "视频错误", f"无法打开视频文件：{os.path.basename(file_path)}")
    #         return
    #
    #     # 显示当前视频文件名（增强用户体验）
    #     if hasattr(self.ui, "label_video_name"):
    #         self.ui.label_video_name.setText(f"当前视频：{os.path.basename(file_path)}")
    #
    #     # 逐帧处理：同时显示原始视频和检测后视频
    #     while self.cap.isOpened():
    #         ret, frame = self.cap.read()
    #         if not ret:
    #             break  # 视频播放结束
    #
    #         # -------------------------- 步骤1：原始视频帧显示到label_ori_video --------------------------
    #         if self.label_ori_video:
    #             # 调整原始帧尺寸（适配标签大小，避免拉伸）
    #             ori_label_size = self.label_ori_video.size()
    #             if ori_label_size.width() > 0 and ori_label_size.height() > 0:
    #                 frame_ori_resized = cv2.resize(frame, (ori_label_size.width(), ori_label_size.height()))
    #             else:
    #                 frame_ori_resized = cv2.resize(frame, (640, 480))  # 默认尺寸
    #
    #             # 格式转换：OpenCV（BGR）→ Qt（RGB）
    #             frame_ori_rgb = cv2.cvtColor(frame_ori_resized, cv2.COLOR_BGR2RGB)
    #             h_ori, w_ori, ch_ori = frame_ori_rgb.shape
    #             bytes_per_line_ori = ch_ori * w_ori  # 每行字节数（Qt图像初始化必需）
    #
    #             # 生成Qt图像并显示
    #             qt_img_ori = QImage(frame_ori_rgb.data, w_ori, h_ori, bytes_per_line_ori, QImage.Format.Format_RGB888)
    #             qt_pix_ori = QPixmap.fromImage(qt_img_ori)
    #             # 保持宽高比缩放，避免画面变形
    #             scaled_pix_ori = qt_pix_ori.scaled(
    #                 self.label_ori_video.size(),
    #                 Qt.AspectRatioMode.KeepAspectRatio,
    #                 Qt.TransformationMode.SmoothTransformation  # 平滑缩放，减少锯齿
    #             )
    #             self.label_ori_video.setPixmap(scaled_pix_ori)
    #         # ----------------------------------------------------------------------------------
    #
    #         # -------------------------- 步骤2：YOLO检测+检测帧显示到label_treated --------------------------
    #         # 1. YOLO检测塑料袋（仅检测26=手提包、41=购物袋，过滤低置信度结果）
    #         results = self.yolo_model(frame, classes=[26, 41], conf=0.3)
    #         annotated_frame = results[0].plot(line_width=2, font_size=12)  # 绘制检测框和类别标签
    #
    #         # 2. 标记是否检测到塑料袋
    #         if len(results[0].boxes) > 0:
    #             self.detected_plastic_bag = True
    #
    #         # 3. 检测帧显示到label_treated
    #         if self.label_treated:
    #             # 调整检测帧尺寸（与原始帧标签尺寸一致，视觉统一）
    #             treated_label_size = self.label_treated.size()
    #             if treated_label_size.width() > 0 and treated_label_size.height() > 0:
    #                 frame_treated_resized = cv2.resize(annotated_frame,
    #                                                    (treated_label_size.width(), treated_label_size.height()))
    #             else:
    #                 frame_treated_resized = cv2.resize(annotated_frame, (640, 480))  # 默认尺寸
    #
    #             # 格式转换：OpenCV（BGR）→ Qt（RGB）
    #             frame_treated_rgb = cv2.cvtColor(frame_treated_resized, cv2.COLOR_BGR2RGB)
    #             h_treated, w_treated, ch_treated = frame_treated_rgb.shape
    #             bytes_per_line_treated = ch_treated * w_treated
    #
    #             # 生成Qt图像并显示
    #             qt_img_treated = QImage(frame_treated_rgb.data, w_treated, h_treated, bytes_per_line_treated,
    #                                     QImage.Format.Format_RGB888)
    #             qt_pix_treated = QPixmap.fromImage(qt_img_treated)
    #             # 保持宽高比缩放
    #             scaled_pix_treated = qt_pix_treated.scaled(
    #                 self.label_treated.size(),
    #                 Qt.AspectRatioMode.KeepAspectRatio,
    #                 Qt.TransformationMode.SmoothTransformation
    #             )
    #             self.label_treated.setPixmap(scaled_pix_treated)
    #         # ----------------------------------------------------------------------------------
    #
    #         # 处理UI事件，避免界面卡顿（单线程下必需）
    #         QApplication.processEvents()
    #
    #     # 视频处理结束：释放资源+弹出检测结果
    #     self.cap.release()
    #     if self.detected_plastic_bag:
    #         QMessageBox.warning(
    #             self,
    #             "检测警告",
    #             f"在视频《{os.path.basename(file_path)}》中检测到塑料袋！"
    #         )
    #     else:
    #         QMessageBox.information(
    #             self,
    #             "检测结果",
    #             f"在视频《{os.path.basename(file_path)}》中未检测到塑料袋。"
    #         )
    def open_and_detect_video(self):
        """打开多格式视频文件并检测塑料袋（修复：恢复视频显示逻辑）"""
        video_formats = "所有视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.rmvb " \
                        "*.mpg *.mpeg *.3gp *.webm *.ts *.m4v *.f4v *.asf);;" \
                        "MP4格式 (*.mp4);;AVI格式 (*.avi);;MOV格式 (*.mov);;MKV格式 (*.mkv);;FLV格式 (*.flv);;" \
                        "所有文件 (*.*)"

        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="选择视频文件（支持多格式）",
            directory="",
            filter=video_formats
        )
        if not file_path:
            return
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "文件错误", f"所选文件不存在：{file_path}")
            return

        # 释放旧资源+清空标签残留画面
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            if self.label_ori_video:
                self.label_ori_video.clear()
            if self.label_treated:
                self.label_treated.clear()

        # 初始化检测标记和最高置信度
        self.detected_plastic_bag = False
        max_confidence = 0.0

        # 打开视频文件
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "视频错误", f"无法打开视频文件：{os.path.basename(file_path)}")
            return
        if hasattr(self.ui, "label_video_name"):
            self.ui.label_video_name.setText(f"当前视频：{os.path.basename(file_path)}")

        # 逐帧检测+显示（核心：补全帧处理与显示逻辑）
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break  # 视频播放结束

            # -------------------------- 修复1：原始帧显示到label_ori_video --------------------------
            if self.label_ori_video:
                # 1. 适配标签尺寸：按label_ori_video大小缩放帧（避免拉伸/超出）
                ori_label_size = self.label_ori_video.size()
                # 若标签有实际尺寸，按标签尺寸缩放；否则用默认尺寸（640x480）
                if ori_label_size.width() > 0 and ori_label_size.height() > 0:
                    frame_ori_resized = cv2.resize(frame, (ori_label_size.width(), ori_label_size.height()))
                else:
                    frame_ori_resized = cv2.resize(frame, (640, 480))

                # 2. 格式转换：OpenCV默认BGR → Qt需要的RGB（否则颜色错乱）
                frame_ori_rgb = cv2.cvtColor(frame_ori_resized, cv2.COLOR_BGR2RGB)
                # 获取帧的宽、高、通道数，计算每行字节数（Qt图像初始化必需）
                h_ori, w_ori, ch_ori = frame_ori_rgb.shape
                bytes_per_line_ori = ch_ori * w_ori

                # 3. 转换为Qt图像并显示到label_ori_video
                qt_img_ori = QImage(
                    frame_ori_rgb.data,  # 帧的像素数据
                    w_ori,  # 帧宽度
                    h_ori,  # 帧高度
                    bytes_per_line_ori,  # 每行字节数
                    QImage.Format.Format_RGB888  # 像素格式（RGB888对应24位色）
                )
                # 转换为QPixmap（QLabel只支持QPixmap显示），并保持宽高比（避免变形）
                qt_pix_ori = QPixmap.fromImage(qt_img_ori).scaled(
                    self.label_ori_video.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,  # 保持宽高比
                    Qt.TransformationMode.SmoothTransformation  # 平滑缩放（减少锯齿）
                )
                self.label_ori_video.setPixmap(qt_pix_ori)  # 最终显示到控件
            # ----------------------------------------------------------------------------------

            # 2. YOLO检测（保持原有逻辑：更新最高置信度）
            results = self.yolo_model(frame, classes=[24,26, 41], conf=0.3)
            annotated_frame = results[0].plot(line_width=2, font_size=12)  # 绘制检测框

            # 更新最高置信度
            if len(results[0].boxes) > 0:
                self.detected_plastic_bag = True
                current_frame_max_conf = max(box.conf[0].item() for box in results[0].boxes)
                if current_frame_max_conf > max_confidence:
                    max_confidence = current_frame_max_conf

            # -------------------------- 修复2：检测帧显示到label_treated --------------------------
            if self.label_treated:
                # 1. 适配标签尺寸：与原始帧标签尺寸一致（视觉统一）
                treated_label_size = self.label_treated.size()
                if treated_label_size.width() > 0 and treated_label_size.height() > 0:
                    frame_treated_resized = cv2.resize(annotated_frame,
                                                       (treated_label_size.width(), treated_label_size.height()))
                else:
                    frame_treated_resized = cv2.resize(annotated_frame, (640, 480))

                # 2. 格式转换：BGR → RGB（避免颜色错乱）
                frame_treated_rgb = cv2.cvtColor(frame_treated_resized, cv2.COLOR_BGR2RGB)
                h_treated, w_treated, ch_treated = frame_treated_rgb.shape
                bytes_per_line_treated = ch_treated * w_treated

                # 3. 转换为Qt图像并显示到label_treated
                qt_img_treated = QImage(
                    frame_treated_rgb.data,
                    w_treated,
                    h_treated,
                    bytes_per_line_treated,
                    QImage.Format.Format_RGB888
                )
                qt_pix_treated = QPixmap.fromImage(qt_img_treated).scaled(
                    self.label_treated.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.label_treated.setPixmap(qt_pix_treated)  # 最终显示到控件
            # ----------------------------------------------------------------------------------

            # 处理UI事件（避免界面卡顿，单线程视频播放必需）
            QApplication.processEvents()

        # 视频结束：释放资源+弹出结果
        self.cap.release()
        if self.detected_plastic_bag:
            QMessageBox.warning(
                self,
                "检测警告",
                f"在视频《{os.path.basename(file_path)}》中检测到塑料袋！\n"
                f"最高置信度：{max_confidence:.2f}"
            )
        else:
            QMessageBox.information(
                self,
                "检测结果",
                f"在视频《{os.path.basename(file_path)}》中未检测到塑料袋。\n"
                f"最高置信度：{max_confidence:.2f}"
            )


if __name__ == "__main__":
    # 解决Qt高DPI显示问题（避免高分屏界面模糊）
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    stats = Stats()
    stats.show()
    sys.exit(app.exec())
