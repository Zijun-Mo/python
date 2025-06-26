"""
面部分析引擎模块
包含主要的业务逻辑和处理流程
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

from analysis.expression_analyzer import ExpressionAnalyzer
from analysis.movement_calculator import MovementCalculator
from visualization.visualizers import LandmarkVisualizer, RaiseEyebrowVisualizer
from visualization.expression_visualizer import ExpressionVisualizer
from data.extractor import DataExtractor


class FacialAnalysisEngine:
    """面部分析引擎 - 主要的业务逻辑类"""
    def __init__(self, movement_calculator: MovementCalculator):
        self.expression_analyzer = ExpressionAnalyzer()
        self.landmark_visualizer = LandmarkVisualizer()
        self.movement_calculator = movement_calculator
        self.raise_eyebrow_visualizer = RaiseEyebrowVisualizer()
        self.expression_visualizer = ExpressionVisualizer()
        self.data_extractor = DataExtractor()
    
    def process_frame_for_specific_expression(self, rgb_frame: np.ndarray, detection_result, 
                                             target_expression: str, reference_landmarks=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        expression_to_ratio = {
            '抬眉': 'raise_eyebrow',
            '闭眼': 'blink',
            '皱鼻': 'sneer',  
            '咧嘴笑': 'smile',
            '撅嘴': 'pucker'
        }
        # 处理单帧图像，只标注特定表情
        annotated_frame = rgb_frame.copy()
        # 先绘制对齐后的参考特征点
        if reference_landmarks and detection_result.face_landmarks:
            # 对齐参考特征点到当前帧
            from core.landmark_processor import LandmarkProcessor
            aligned_reference = LandmarkProcessor.align_landmarks(reference_landmarks, detection_result.face_landmarks[0])
            # 绘制参考特征点
            annotated_frame = self.landmark_visualizer.draw(annotated_frame, aligned_reference, color=(0,255,0))
        # 再绘制当前帧特征点
        annotated_frame = self.landmark_visualizer.draw(annotated_frame, detection_result)
        # 标注耸鼻子检测的特征点
        if detection_result.face_landmarks:
            annotated_frame = self.raise_eyebrow_visualizer.draw(annotated_frame, detection_result.face_landmarks[0])
        # 分析表情
        expressions = {}
        movement_ratios = None
        if detection_result.face_blendshapes:
            expressions = self.expression_analyzer.analyze_expressions(detection_result, reference_landmarks)
            # 计算运动幅度比率
            if reference_landmarks and detection_result.face_landmarks:
                movement_ratios = self.movement_calculator.calculate_facial_movement_ratios(
                    reference_landmarks, detection_result.face_landmarks[0])
                self.movement_calculator.add_movement_data({expression_to_ratio[target_expression]: movement_ratios[expression_to_ratio[target_expression]]})
            # 只绘制特定表情信息
            annotated_frame = self.expression_visualizer.draw(
                annotated_frame, expressions, target_expression, movement_ratios)
        # 提取数据
        frame_data = {
            'landmarks': self.data_extractor.extract_landmarks_data(detection_result),
            'blendshapes': self.data_extractor.extract_blendshapes_data(detection_result),
            'expressions': expressions,
            'movement_ratios': movement_ratios
        }
        return annotated_frame, frame_data
