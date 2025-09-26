from PyQt6.QtWidgets import (QApplication, QWidget, QFileDialog,
                             QMessageBox, QCheckBox, QLabel)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from PyQt6.uic import loadUi
import sys
import os
import cv2
from models.experimental import attempt_load
from utils.torch_utils import select_device
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
        # -------------------------- 关键修改1：加载双模型（塑料袋+头盔检测） --------------------------
        try:
            # 1. 原塑料袋检测模型（yolov8n.pt，检测24=手提箱、26=手提包、41=购物袋）
            self.yolo_bag_model = YOLO('yolov8n.pt')
            # 2. 新增头盔检测模型（helmet_head_person_s.pt，需确保路径正确）
            # 若权重在项目根目录，直接写文件名；否则写绝对路径（如"D:/yolov5/helmet_head_person_s.pt"）
            self.yolo_helmet_model = YOLO('D:\yolov5\yolov5-master/helmet_head_person_s.pt')
            print("双YOLO模型加载成功（塑料袋+头盔检测）")
        except Exception as e:
            QMessageBox.critical(self, "模型错误", f"YOLO模型加载失败：{str(e)}\n请检查helmet_head_person_s.pt路径")
            sys.exit(1)

        # 2. 初始化视频捕获对象和检测标记（新增头盔相关标记）
        self.cap = None
        self.detected_plastic_bag = False  # 塑料袋检测标记
        self.detected_helmet = False       # 头盔检测标记
        self.detected_head = False         # 人头检测标记
        self.detected_person = False       # 人体检测标记

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

    def open_and_detect_video(self):
        """打开多格式视频文件并检测（塑料袋+头盔/人头/人体，合并显示结果）"""
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

        # 初始化检测标记和最高置信度（新增头盔相关置信度）
        self.detected_plastic_bag = False
        self.detected_helmet = False
        self.detected_head = False
        self.detected_person = False
        max_bag_conf = 0.0       # 塑料袋最高置信度
        max_helmet_conf = 0.0    # 头盔最高置信度
        max_head_conf = 0.0      # 人头最高置信度
        max_person_conf = 0.0    # 人体最高置信度

        # 打开视频文件
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "视频错误", f"无法打开视频文件：{os.path.basename(file_path)}")
            return
        if hasattr(self.ui, "label_video_name"):
            self.ui.label_video_name.setText(f"当前视频：{os.path.basename(file_path)}")

        # 逐帧检测+显示（核心：双模型检测+合并结果）
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break  # 视频播放结束

            # -------------------------- 1. 原始帧显示到label_ori_video（保持不变） --------------------------
            if self.label_ori_video:
                ori_label_size = self.label_ori_video.size()
                if ori_label_size.width() > 0 and ori_label_size.height() > 0:
                    frame_ori_resized = cv2.resize(frame, (ori_label_size.width(), ori_label_size.height()))
                else:
                    frame_ori_resized = cv2.resize(frame, (640, 480))

                frame_ori_rgb = cv2.cvtColor(frame_ori_resized, cv2.COLOR_BGR2RGB)
                h_ori, w_ori, ch_ori = frame_ori_rgb.shape
                bytes_per_line_ori = ch_ori * w_ori

                qt_img_ori = QImage(
                    frame_ori_rgb.data, w_ori, h_ori, bytes_per_line_ori,
                    QImage.Format.Format_RGB888
                )
                qt_pix_ori = QPixmap.fromImage(qt_img_ori).scaled(
                    self.label_ori_video.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.label_ori_video.setPixmap(qt_pix_ori)
            # ----------------------------------------------------------------------------------

            # -------------------------- 关键修改2：双模型检测（塑料袋+头盔/人头/人体） --------------------------
            # 1. 塑料袋检测（原逻辑不变，检测24=手提箱、26=手提包、41=购物袋）
            bag_results = self.yolo_bag_model(frame, classes=[24,26, 41], conf=0.3)
            # 2. 头盔/人头/人体检测（用helmet_head_person_s.pt，默认检测所有类别，conf=0.3过滤低置信度）
            # 假设你的权重类别顺序：0=person（人体）、1=head（人头）、2=helmet（头盔）（若不同需调整）
            helmet_results = self.yolo_helmet_model(frame, conf=0.3)

            # 3. 合并检测结果：先绘制塑料袋检测框，再绘制头盔相关检测框（避免覆盖）
            # 复制原始帧作为基础，分别叠加两个模型的检测框
            merged_frame = frame.copy()
            # 叠加塑料袋检测框（用默认颜色）
            merged_frame = bag_results[0].plot(line_width=2, font_size=12, img=merged_frame)
            # 叠加头盔相关检测框（可自定义颜色，避免与塑料袋框混淆，如头盔=绿色、人头=红色、人体=蓝色）
            merged_frame = helmet_results[0].plot(
                line_width=2,
                font_size=12,
                img=merged_frame,
                colors=[(255,0,0), (0,0,255), (0,255,0)]  # 类别颜色：0=person(蓝)、1=head(红)、2=helmet(绿)
            )
            # ----------------------------------------------------------------------------------

            # -------------------------- 关键修改3：更新检测标记和最高置信度（双模型） --------------------------
            # 更新塑料袋检测标记和置信度
            if len(bag_results[0].boxes) > 0:
                self.detected_plastic_bag = True
                current_bag_conf = max(box.conf[0].item() for box in bag_results[0].boxes)
                if current_bag_conf > max_bag_conf:
                    max_bag_conf = current_bag_conf

            # 更新头盔/人头/人体检测标记和置信度（需匹配权重的类别顺序）
            for box in helmet_results[0].boxes:
                cls = int(box.cls[0].item())  # 获取类别索引
                conf = box.conf[0].item()     # 获取置信度
                # 假设类别顺序：0=person（人体）、1=head（人头）、2=helmet（头盔）（若你的权重类别顺序不同，需修改cls对应的含义）
                if cls == 0:  # 人体
                    self.detected_person = True
                    if conf > max_person_conf:
                        max_person_conf = conf
                elif cls == 1:  # 人头
                    self.detected_head = True
                    if conf > max_head_conf:
                        max_head_conf = conf
                elif cls == 2:  # 头盔
                    self.detected_helmet = True
                    if conf > max_helmet_conf:
                        max_helmet_conf = conf
            # ----------------------------------------------------------------------------------

            # -------------------------- 4. 合并检测帧显示到label_treated（用merged_frame） --------------------------
            if self.label_treated:
                treated_label_size = self.label_treated.size()
                if treated_label_size.width() > 0 and treated_label_size.height() > 0:
                    frame_treated_resized = cv2.resize(merged_frame,
                                                       (treated_label_size.width(), treated_label_size.height()))
                else:
                    frame_treated_resized = cv2.resize(merged_frame, (640, 480))

                frame_treated_rgb = cv2.cvtColor(frame_treated_resized, cv2.COLOR_BGR2RGB)
                h_treated, w_treated, ch_treated = frame_treated_rgb.shape
                bytes_per_line_treated = ch_treated * w_treated

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
                self.label_treated.setPixmap(qt_pix_treated)
            # ----------------------------------------------------------------------------------

            # 处理UI事件（避免界面卡顿，单线程视频播放必需）
            QApplication.processEvents()

        # -------------------------- 关键修改4：检测结果提示（同时显示双模型结果） --------------------------
        # 释放资源
        self.cap.release()
        # 构建结果提示文本
        result_text = f"视频《{os.path.basename(file_path)}》检测结果：\n"
        # 塑料袋结果
        if self.detected_plastic_bag:
            result_text += f"- 塑料袋/手提包：已检测到（最高置信度：{max_bag_conf:.2f}）\n"
        else:
            result_text += f"- 塑料袋/手提包：未检测到\n"
        # 头盔相关结果
        result_text += f"- 人体：{'已检测到（最高置信度：{max_person_conf:.2f}）' if self.detected_person else '未检测到'}\n"
        result_text += f"- 人头：{'已检测到（最高置信度：{max_head_conf:.2f}）' if self.detected_head else '未检测到'}\n"
        result_text += f"- 头盔：{'已检测到（最高置信度：{max_helmet_conf:.2f}）' if self.detected_helmet else '未检测到'}\n"

        # 弹出结果（根据是否有风险目标选择警告/信息框）
        if self.detected_plastic_bag:
            QMessageBox.warning(self, "检测警告", result_text)
        else:
            QMessageBox.information(self, "检测结果", result_text)
        # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    # 解决Qt高DPI显示问题（避免高分屏界面模糊）
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    stats = Stats()
    stats.show()
    sys.exit(app.exec())