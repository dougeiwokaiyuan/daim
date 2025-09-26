# 务必在代码最开头添加YOLOv5路径（否则找不到models模块）
import sys
import os
# 1. 替换成你的YOLOv5项目根目录（必须包含models、utils文件夹）
yolov5_root = "D:\\yolov5\\yolov5-master"
if not os.path.exists(yolov5_root):
    print(f"错误：YOLOv5路径不存在！请检查：{yolov5_root}")
    sys.exit(1)
if yolov5_root not in sys.path:
    sys.path.append(yolov5_root)  # 加入Python搜索路径

# 2. 导入依赖（YOLOv5原生模块 + YOLOv8库 + PyQt库）
from models.experimental import attempt_load  # YOLOv5原生加载函数
from utils.torch_utils import select_device    # YOLOv5设备选择
from utils.general import non_max_suppression, scale_boxes  # YOLOv5检测后处理
from utils.plots import Annotator  # YOLOv5绘制检测框
from ultralytics import YOLO  # YOLOv8加载库
from PyQt6.QtWidgets import (QApplication, QWidget, QFileDialog, QMessageBox, QCheckBox, QLabel)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from PyQt6.uic import loadUi
import cv2
import torch
import numpy as np

# 关闭YOLO调试信息
os.environ['YOLO_VERBOSE'] = 'False'


class Stats(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = loadUi("D:/qtdesigner/main11111.ui", self)
        self.init_components()
        self.bind_events()

    def init_components(self):
        """初始化双模型（YOLOv5头盔检测 + YOLOv8塑料袋检测）"""
        try:
            # -------------------------- 1. 初始化设备（GPU/CPU，双模型共用） --------------------------
            self.device = select_device('0')  # '0'=GPU，'cpu'=CPU（无GPU时修改）
            self.half = self.device.type != 'cpu'  # 半精度推理（GPU支持，加速检测）

            # -------------------------- 2. 加载YOLOv5模型（helmet_head_person_s.pt） --------------------------
            # 替换成你的头盔权重路径（相对/绝对路径均可）
            helmet_weights = "helmet_head_person_s.pt"  # 若在代码目录，直接写文件名
            # 加载权重（attempt_load是YOLOv5原生函数，自动处理单/多权重文件）
            self.yolo5_helmet_model = attempt_load(helmet_weights, map_location=self.device)
            # 半精度优化（GPU时启用，提升速度）
            if self.half:
                self.yolo5_helmet_model.half()
            # 获取YOLOv5模型的类别名（假设你的权重类别：0=person,1=head,2=helmet，可根据实际修改）
            self.yolo5_names = ['person', 'head', 'helmet']

            # -------------------------- 3. 加载YOLOv8模型（yolov8n.pt，塑料袋检测） --------------------------
            self.yolo8_bag_model = YOLO('yolov8n.pt')
            # 塑料袋检测的目标类别（COCO数据集：24=手提箱，26=手提包，41=购物袋）
            self.yolo8_bag_classes = [24, 26, 41]

            print("双模型加载成功！\n- YOLOv5：helmet_head_person_s.pt（头盔/人头/人体）\n- YOLOv8：yolov8n.pt（塑料袋）")

        except Exception as e:
            QMessageBox.critical(self, "模型错误", f"加载失败：{str(e)}\n请检查：1. YOLOv5路径 2. 权重文件 3. 设备是否可用")
            sys.exit(1)

        # 初始化其他变量（视频捕获、检测标记等）
        self.cap = None
        self.detected = {
            'person': False, 'head': False, 'helmet': False, 'bag': False,
            'max_conf': {'person': 0, 'head': 0, 'helmet': 0, 'bag': 0}
        }

        # 获取UI控件（保持原逻辑）
        self.label_ori_video = getattr(self.ui, "label_ori_video", None)
        self.label_treated = getattr(self.ui, "label_treated", None)
        if not self.label_ori_video or not self.label_treated:
            QMessageBox.warning(self, "控件警告", "未找到视频显示标签，请检查UI控件名")

    def bind_events(self):
        """绑定事件（保持原逻辑）"""
        if hasattr(self.ui, "videoBtn"):
            self.ui.videoBtn.clicked.connect(self.open_and_detect_video)
        if hasattr(self.ui, "detect_plastic_bag_intrusion"):
            self.ui.detect_plastic_bag_intrusion.stateChanged.connect(self.handle_foreign_invasion)

    def handle_foreign_invasion(self, state):
        """异物入侵复选框逻辑（保持原逻辑）"""
        if state == Qt.CheckState.Checked.value:
            self.open_and_detect_video()
            if hasattr(self.ui, "detect_plastic_bag_intrusion"):
                self.ui.detect_plastic_bag_intrusion.setChecked(False)

    def open_and_detect_video(self):
        """核心：双模型同步检测（YOLOv5头盔 + YOLOv8塑料袋）"""
        # 1. 选择视频文件（保持原逻辑）
        video_formats = "所有视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", video_formats)
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "文件错误", "视频文件不存在或未选择")
            return

        # 2. 释放旧资源（保持原逻辑）
        if self.cap and self.cap.isOpened():
            self.cap.release()
            if self.label_ori_video:
                self.label_ori_video.clear()
            if self.label_treated:
                self.label_treated.clear()
        self.detected = {'person': False, 'head': False, 'helmet': False, 'bag': False,
                         'max_conf': {'person': 0, 'head': 0, 'helmet': 0, 'bag': 0}}

        # 3. 打开视频
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "视频错误", f"无法打开视频：{os.path.basename(file_path)}")
            return
        if hasattr(self.ui, "label_video_name"):
            self.ui.label_video_name.setText(f"当前视频：{os.path.basename(file_path)}")

        # 4. 逐帧检测（核心：双模型并行处理）
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break  # 视频结束

            # -------------------------- 步骤1：显示原始视频（保持原逻辑） --------------------------
            if self.label_ori_video:
                ori_resized = cv2.resize(frame, (self.label_ori_video.width(), self.label_ori_video.height()))
                ori_rgb = cv2.cvtColor(ori_resized, cv2.COLOR_BGR2RGB)
                qt_ori = QImage(ori_rgb.data, ori_rgb.shape[1], ori_rgb.shape[0], QImage.Format.Format_RGB888)
                self.label_ori_video.setPixmap(QPixmap.fromImage(qt_ori))

            # -------------------------- 步骤2：YOLOv5检测（头盔/人头/人体） --------------------------
            # 预处理帧（适配YOLOv5输入格式）
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB
            img = cv2.resize(img, (640, 640))  # 缩放为YOLOv5默认输入尺寸（可根据权重调整）
            img = img.transpose((2, 0, 1))  # HWC→CHW（PyTorch输入格式）
            img = np.ascontiguousarray(img)  # 内存连续化，提升效率
            img = torch.from_numpy(img).to(self.device)  # 转Tensor并传入设备
            img = img.half() if self.half else img.float()  # 半精度/单精度
            img /= 255.0  # 归一化（0-255→0-1）
            if img.ndimension() == 3:
                img = img.unsqueeze(0)  # 增加批次维度（1,3,640,640）

            # 模型推理 + 后处理（非极大值抑制，过滤重复框）
            with torch.no_grad():  # 禁用梯度计算，提升速度
                pred = self.yolo5_helmet_model(img)[0]  # 推理结果
                # 非极大值抑制（conf_thres=0.3：过滤低置信度，iou_thres=0.45：合并重叠框）
                pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45, max_det=1000)

            # 绘制YOLOv5检测框（用Annotator工具，适配原帧尺寸）
            annotator = Annotator(frame.copy(), line_width=2, font_size=12)
            for det in pred:  # 遍历每一个检测结果
                if len(det):
                    # 缩放检测框到原帧尺寸（因为推理时用了640尺寸，需映射回原帧）
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                    # 遍历每个检测框，记录类别和置信度
                    for *xyxy, conf, cls in reversed(det):
                        cls_int = int(cls)
                        cls_name = self.yolo5_names[cls_int] if cls_int < len(self.yolo5_names) else f"class_{cls_int}"
                        # 更新检测标记和最高置信度
                        if cls_name in self.detected:
                            self.detected[cls_name] = True
                            if conf > self.detected['max_conf'][cls_name]:
                                self.detected['max_conf'][cls_name] = round(float(conf), 2)
                        # 绘制框和标签（颜色：person=蓝(255,0,0)，head=红(0,0,255)，helmet=绿(0,255,0)）
                        color = (255, 0, 0) if cls_name == 'person' else (0, 0, 255) if cls_name == 'head' else (0, 255, 0)
                        annotator.box_label(xyxy, f"{cls_name} {conf:.2f}", color=color)

            # 此时annotator.result() 是已绘制YOLOv5检测框的帧
            frame_with_yolov5 = annotator.result()

            # -------------------------- 步骤3：YOLOv8检测（塑料袋） --------------------------
            # 用YOLOv8推理，指定检测类别（24=手提箱，26=手提包，41=购物袋）
            yolov8_results = self.yolo8_bag_model(frame_with_yolov5, classes=self.yolo8_bag_classes, conf=0.3)
            # 绘制YOLOv8检测框（直接在YOLOv5的结果帧上叠加，颜色默认橙色，避免冲突）
            frame_final = yolov8_results[0].plot(line_width=2, font_size=12, img=frame_with_yolov5)

            # 更新塑料袋检测标记和最高置信度
            if len(yolov8_results[0].boxes) > 0:
                self.detected['bag'] = True
                max_bag_conf = max(float(box.conf) for box in yolov8_results[0].boxes)
                if max_bag_conf > self.detected['max_conf']['bag']:
                    self.detected['max_conf']['bag'] = round(max_bag_conf, 2)

            # -------------------------- 步骤4：显示最终检测结果（双模型叠加） --------------------------
            if self.label_treated:
                treated_resized = cv2.resize(frame_final, (self.label_treated.width(), self.label_treated.height()))
                treated_rgb = cv2.cvtColor(treated_resized, cv2.COLOR_BGR2RGB)
                qt_treated = QImage(treated_rgb.data, treated_rgb.shape[1], treated_rgb.shape[0], QImage.Format.Format_RGB888)
                self.label_treated.setPixmap(QPixmap.fromImage(qt_treated))

            # 处理UI事件，避免卡顿
            QApplication.processEvents()

        # -------------------------- 步骤5：检测结束，显示结果 --------------------------
        self.cap.release()
        result_text = f"视频《{os.path.basename(file_path)}》检测结果：\n" \
                      f"- 人体：{'已检测到（最高置信度：{:.2f}）'.format(self.detected['max_conf']['person']) if self.detected['person'] else '未检测到'}\n" \
                      f"- 人头：{'已检测到（最高置信度：{:.2f}）'.format(self.detected['max_conf']['head']) if self.detected['head'] else '未检测到'}\n" \
                      f"- 头盔：{'已检测到（最高置信度：{:.2f}）'.format(self.detected['max_conf']['helmet']) if self.detected['helmet'] else '未检测到'}\n" \
                      f"- 塑料袋/手提包：{'已检测到（最高置信度：{:.2f}）'.format(self.detected['max_conf']['bag']) if self.detected['bag'] else '未检测到'}"
        QMessageBox.warning(self, "检测警告", result_text) if self.detected['bag'] else QMessageBox.information(self, "检测结果", result_text)


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    stats = Stats()
    stats.show()
    sys.exit(app.exec())