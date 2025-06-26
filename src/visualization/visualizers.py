"""
可视化器基类和具体实现
包含特征点、表情信息等可视化功能
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from PIL import Image, ImageDraw
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

from .font_manager import FontManager
# 导入LandmarkProcessor以获取RPY角度信息
from core.landmark_processor import LandmarkProcessor


class BaseVisualizer(ABC):
    """可视化器基类"""
    
    def __init__(self):
        self.font_manager = FontManager()
    
    @abstractmethod
    def draw(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """绘制方法，需要子类实现"""
        pass


class LandmarkVisualizer(BaseVisualizer):
    """特征点可视化器"""
    
    def __init__(self):
        super().__init__()
        self.landmark_processor = LandmarkProcessor()
    
    def draw(self, rgb_image: np.ndarray, detection_result, color=None) -> np.ndarray:
        """绘制人脸特征点，支持自定义颜色，且始终绘制mesh"""
        try: 
            face_landmarks = detection_result.face_landmarks[0]
        except Exception as e:
            face_landmarks = detection_result
        annotated_image = np.copy(rgb_image)

        # 先画mesh
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in face_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
        # 再画点（如有颜色要求）
        if color is not None:
            h, w = annotated_image.shape[:2]
            for landmark in face_landmarks:
                x = int(landmark.x * w) if landmark.x <= 1 else int(landmark.x)
                y = int(landmark.y * h) if landmark.y <= 1 else int(landmark.y)
                cv2.circle(annotated_image, (x, y), 2, color, -1)
        # 为第一个检测到的人脸添加RPY角度显示
        if face_landmarks is not None:
            self._draw_rpy_angles_cv2(annotated_image, face_landmarks)
        return annotated_image
    
    def _draw_rpy_angles_cv2(self, image, landmarks):
        """使用OpenCV在图像右上角绘制RPY角度信息"""
        try:
            # 获取面部朝向角度
            orientation = self.landmark_processor.detect_face_orientation(landmarks)
            
            h, w = image.shape[:2]
            
            # 右上角位置
            x_start = w - 220
            y_start = 30
            
            # 绘制半透明背景
            overlay = image.copy()
            cv2.rectangle(overlay, (x_start - 10, y_start - 20), 
                         (x_start + 200, y_start + 70), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # 绘制边框
            cv2.rectangle(image, (x_start - 10, y_start - 20), 
                         (x_start + 200, y_start + 70), (255, 255, 255), 1)
            
            # 字体设置
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # 绘制标题
            cv2.putText(image, "Face Orientation (RPY)", 
                       (x_start, y_start), font, font_scale, (255, 255, 255), thickness)
            
            # 根据角度大小选择颜色
            yaw_color = (100, 100, 255) if abs(orientation['yaw']) > 15 else (100, 255, 100)
            pitch_color = (100, 100, 255) if abs(orientation['pitch']) > 15 else (100, 255, 100)
            roll_color = (100, 100, 255) if abs(orientation['roll']) > 15 else (100, 255, 100)
            
            # 绘制角度值
            cv2.putText(image, f"Yaw:   {orientation['yaw']:6.1f}deg", 
                       (x_start, y_start + 18), font, font_scale, yaw_color, thickness)
            cv2.putText(image, f"Pitch: {orientation['pitch']:6.1f}deg", 
                       (x_start, y_start + 35), font, font_scale, pitch_color, thickness)
            cv2.putText(image, f"Roll:  {orientation['roll']:6.1f}deg", 
                       (x_start, y_start + 52), font, font_scale, roll_color, thickness)
            
        except Exception as e:
            # 如果计算失败，显示错误信息
            h, w = image.shape[:2]
            x_start = w - 180
            y_start = 30
            
            cv2.rectangle(image, (x_start - 10, y_start - 20), 
                         (x_start + 170, y_start + 40), (0, 0, 100), -1)
            cv2.rectangle(image, (x_start - 10, y_start - 20), 
                         (x_start + 170, y_start + 40), (100, 100, 255), 1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, "RPY Calc Failed", 
                       (x_start, y_start), font, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f"Error: {str(e)[:12]}...", 
                       (x_start, y_start + 20), font, 0.4, (200, 200, 255), 1)


class RaiseEyebrowVisualizer(BaseVisualizer):
    """抬眉毛特征点可视化器"""
    
    def __init__(self):
        super().__init__()
        self.landmark_processor = LandmarkProcessor()
    
    def draw(self, image: np.ndarray, landmarks) -> np.ndarray:
        """在图像上标注抬眉毛检测使用的四个特征点"""
        if not landmarks:
            return image
        
        # 抬眉毛检测使用的四个点
        eyebrow_points = [105, 52, 334, 282]  # 左眉外侧、左眉内侧、右眉外侧、右眉内侧
        point_names = ['左眉外侧', '左眉内侧', '右眉外侧', '右眉内侧']
        
        # 转换为PIL图像以支持中文标注
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        font = self.font_manager.get_chinese_font(12)
        h, w = image.shape[:2]
        
        # 记录点的坐标用于绘制
        point_coords = []
        
        # 绘制特征点
        for i, point_idx in enumerate(eyebrow_points):
            if point_idx < len(landmarks):
                landmark = landmarks[point_idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                point_coords.append((x, y))
                
                # 绘制圆圈标记点
                color = (255, 255, 0) if i < 2 else (0, 255, 255)  # 左侧黄色，右侧青色
                draw.ellipse([x-4, y-4, x+4, y+4], fill=color, outline=(255, 255, 255), width=2)
                
                # 添加文字标注
                text = f"{point_idx}:{point_names[i]}"
                text_x = x + 10
                text_y = y - 15
                draw.text((text_x, text_y), text, font=font, fill=color)
        
        # 绘制眉毛中点并标注Y坐标
        if len(point_coords) == 4:
            self._draw_eyebrow_midpoints(draw, point_coords, landmarks, font, h)
        
        # 转换回OpenCV格式
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_image
    
    def _draw_eyebrow_midpoints(self, draw, point_coords, landmarks, font, h):
        """绘制眉毛中点并显示相关信息"""
        # 计算左右眉毛中点
        left_mid_x = (point_coords[0][0] + point_coords[1][0]) // 2
        left_mid_y = (point_coords[0][1] + point_coords[1][1]) // 2
        
        right_mid_x = (point_coords[2][0] + point_coords[3][0]) // 2
        right_mid_y = (point_coords[2][1] + point_coords[3][1]) // 2
        
        # 绘制中点
        draw.ellipse([left_mid_x-5, left_mid_y-5, left_mid_x+5, left_mid_y+5], fill=(255, 0, 255), outline=(255, 255, 255))
        draw.ellipse([right_mid_x-5, right_mid_y-5, right_mid_x+5, right_mid_y+5], fill=(255, 0, 255), outline=(255, 255, 255))

        # 获取标准化的Y坐标
        left_brow_landmark_1 = landmarks[105]
        left_brow_landmark_2 = landmarks[52]
        right_brow_landmark_1 = landmarks[334]
        right_brow_landmark_2 = landmarks[282]

        left_mid_y_norm = (left_brow_landmark_1.y + left_brow_landmark_2.y) / 2
        right_mid_y_norm = (right_brow_landmark_1.y + right_brow_landmark_2.y) / 2

        # 标注Y坐标
        left_text = f"Y: {left_mid_y_norm:.3f}"
        right_text = f"Y: {right_mid_y_norm:.3f}"
        draw.text((left_mid_x + 10, left_mid_y - 10), left_text, font=font, fill=(255, 255, 0))
        draw.text((right_mid_x + 10, right_mid_y - 10), right_text, font=font, fill=(0, 255, 255))

        # 底部统计信息
        self._draw_statistics(draw, left_mid_y_norm, right_mid_y_norm, h, font)

    def _draw_statistics(self, draw, left_y, right_y, h, font):
        """绘制底部统计信息"""
        # Y坐标差异
        y_diff = abs(left_y - right_y)
        
        draw.text((20, h-55), f"左眉中点 Y: {left_y:.3f}", font=font, fill=(255, 255, 0))
        draw.text((20, h-40), f"右眉中点 Y: {right_y:.3f}", font=font, fill=(0, 255, 255))
        draw.text((20, h-25), f"Y坐标差异: {y_diff:.4f}", font=font, fill=(255, 255, 255))
