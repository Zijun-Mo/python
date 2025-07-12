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
from analysis.synkinesis_calculator import SynkinesisCalculator


class FacialAnalysisEngine:
    """面部分析引擎 - 主要的业务逻辑类"""
    def __init__(self, movement_calculator: MovementCalculator, synkinesis_calculator: SynkinesisCalculator):
        self.expression_analyzer = ExpressionAnalyzer()
        self.landmark_visualizer = LandmarkVisualizer()
        self.movement_calculator = movement_calculator
        self.synkinesis_calculator = synkinesis_calculator
        self.raise_eyebrow_visualizer = RaiseEyebrowVisualizer()
        self.expression_visualizer = ExpressionVisualizer()
        self.data_extractor = DataExtractor()

    def process_frame_for_specific_expression(self, rgb_frame: np.ndarray, detection_result, 
                                             target_expression: str, reference_result = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        expression_to_ratio = {
            '抬眉': 'raise_eyebrow',
            '闭眼': 'blink',
            '皱鼻': 'sneer',  
            '咧嘴笑': 'smile',
            '撅嘴': 'pucker'
        }
        # 处理单帧图像，只标注特定表情
        annotated_frame = rgb_frame.copy()
        # 再绘制当前帧特征点
        annotated_frame = self.landmark_visualizer.draw(annotated_frame, detection_result)
        # 标注耸鼻子检测的特征点
        if detection_result.face_landmarks:
            annotated_frame = self.raise_eyebrow_visualizer.draw(annotated_frame, detection_result.face_landmarks[0])
        # 分析表情
        expressions = {}
        movement_ratios = None
        if detection_result.face_blendshapes:
            expressions = self.expression_analyzer.analyze_expressions(detection_result, reference_result.face_landmarks[0])
            synkinesis_scores = self.synkinesis_calculator.calculate_facial_movement(
                reference_result.face_blendshapes[0], detection_result.face_blendshapes[0], target_expression
            )
            self.synkinesis_calculator.add_movement_data(synkinesis_scores, target_expression)
            # 计算运动幅度比率
            if reference_result.face_landmarks and detection_result.face_landmarks:
                movement_ratios = self.movement_calculator.calculate_facial_movement_ratios(
                    reference_result.face_landmarks[0], detection_result.face_landmarks[0])
                self.movement_calculator.add_movement_data({expression_to_ratio[target_expression]: movement_ratios[expression_to_ratio[target_expression]]})
            # 只绘制特定表情信息
            annotated_frame = self.expression_visualizer.draw(
                annotated_frame, expressions, target_expression, movement_ratios, synkinesis_scores)
        # 提取数据
        frame_data = {
            'landmarks': self.data_extractor.extract_landmarks_data(detection_result),
            'blendshapes': self.data_extractor.extract_blendshapes_data(detection_result),
            'expressions': expressions,
            'movement_ratios': movement_ratios
        }
        return annotated_frame, frame_data
