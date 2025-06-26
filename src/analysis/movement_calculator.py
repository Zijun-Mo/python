"""
运动幅度计算器
用于计算面部运动的对称性比率
"""

from typing import Dict, List, Tuple
from core.landmark_processor import LandmarkProcessor
import numpy as np
from config import Config


class MovementCalculator:
    """运动幅度计算器"""
    def __calculate_average_movement_ratios(self) -> Dict[str, float]:
        """计算各表情运动幅度的平均值"""
        averages = {}
        for expression, ratios in self.movement_history.items():
            if ratios:
                averages[expression] = sum(ratios) / len(ratios)
            else:
                averages[expression] = 0.0
        return averages
    
    def __calculate_symmetry_scores(self, average_ratios: Dict[str, float] = None) -> Dict[str, int]:
        """
        根据运动幅度比率计算对称评分
        评分标准：
        1分：完全不动（肉眼观察不到的运动）
        2分：运动幅度比率达到正常范围的1%--30%（可参考对侧）
        3分：运动幅度比率达到正常范围的31%--60%（可参考对侧）
        4分：运动幅度比率达到正常范围的61%--90%（可参考对侧）
        5分：双侧面部运动对称91%以上
        """
        if average_ratios is None:
            average_ratios = self.__calculate_average_movement_ratios()
        
        scores = {}
        expression_mapping = {
            "raise_eyebrow": "抬眉",
            "blink": "闭眼",
            "sneer": "皱鼻",
            "smile": "咧嘴笑",
            "pucker": "撅嘴"
        }
        
        for expression, ratio in average_ratios.items():
            if expression in expression_mapping:
                # 将比率转换为百分比
                percentage = ratio * 100
                
                if percentage <= 0.1:  # 几乎不动
                    score = 1
                elif 1 <= percentage <= 30:
                    score = 2
                elif 31 <= percentage <= 60:
                    score = 3
                elif 61 <= percentage <= 90:
                    score = 4
                else:  # 91%以上
                    score = 5
                
                scores[expression_mapping[expression]] = score
        
        return scores
    def __init__(self):
        self.landmark_processor = LandmarkProcessor()
        # 用于累积多帧数据计算平均值
        self.movement_history = {
            "raise_eyebrow": [],
            "blink": [],
            "sneer": [],
            "smile": [],
            "pucker": []
        }
    
    def calculate_facial_movement_ratios(self, rest_landmarks, move_landmarks, visible_threshold: float = Config.VISIBLE_THRESHOLD, correct_orientation: bool = True) -> Dict[str, float]:
        """计算五项运动幅度比率"""
        results = {}
        
        # 如果启用朝向校正，先校正面部朝向
        if correct_orientation:
            rest_rpy = self.landmark_processor.detect_face_orientation(rest_landmarks)
            move_rpy = self.landmark_processor.detect_face_orientation(move_landmarks)
            corrected_rest_landmarks = self.landmark_processor.correct_face_orientation(rest_landmarks, rest_rpy)
            corrected_move_landmarks = self.landmark_processor.correct_face_orientation(move_landmarks, move_rpy)
            
            # 再进行landmarks对齐
            aligned_move_landmarks = self.landmark_processor.align_landmarks(corrected_move_landmarks, corrected_rest_landmarks)
            rest_landmarks = corrected_rest_landmarks
        else:
            # 直接对landmarks进行对齐
            aligned_move_landmarks = self.landmark_processor.align_landmarks(move_landmarks, rest_landmarks)
        
        # 1. 抬眉毛：眉毛中点上下方向移动
        left_brow_rest = (self.landmark_processor.get_point(rest_landmarks, 105) + 
                         self.landmark_processor.get_point(rest_landmarks, 52)) / 2
        left_brow_move = (self.landmark_processor.get_point(aligned_move_landmarks, 105) + 
                         self.landmark_processor.get_point(aligned_move_landmarks, 52)) / 2
        right_brow_rest = (self.landmark_processor.get_point(rest_landmarks, 334) + 
                          self.landmark_processor.get_point(rest_landmarks, 282)) / 2
        right_brow_move = (self.landmark_processor.get_point(aligned_move_landmarks, 334) + 
                          self.landmark_processor.get_point(aligned_move_landmarks, 282)) / 2
        
        left_l = abs(left_brow_move[1] - left_brow_rest[1])
        right_l = abs(right_brow_move[1] - right_brow_rest[1])
        small_l = min(left_l, right_l)
        large_l = max(left_l, right_l)
        results["raise_eyebrow"] = small_l / large_l if large_l > visible_threshold else 0.0
        
        # 2. 轻闭眼：眼裂高低差（改为两点间距离）
        left_eye_rest = self.landmark_processor.calc_distance(
            self.landmark_processor.get_point(rest_landmarks, 159),
            self.landmark_processor.get_point(rest_landmarks, 145))
        left_eye_move = self.landmark_processor.calc_distance(
            self.landmark_processor.get_point(aligned_move_landmarks, 159),
            self.landmark_processor.get_point(aligned_move_landmarks, 145))
        right_eye_rest = self.landmark_processor.calc_distance(
            self.landmark_processor.get_point(rest_landmarks, 374),
            self.landmark_processor.get_point(rest_landmarks, 386))
        right_eye_move = self.landmark_processor.calc_distance(
            self.landmark_processor.get_point(aligned_move_landmarks, 374),
            self.landmark_processor.get_point(aligned_move_landmarks, 386))
        left_l = left_eye_rest - left_eye_move
        right_l = right_eye_rest - right_eye_move
        small_l = min(left_l, right_l)
        large_l = max(left_l, right_l)
        results["blink"] = small_l / large_l if large_l > visible_threshold else 0.0
        
        # 3. 耸鼻子：鼻翼根至内眦的距离变短
        left_rest = self.landmark_processor.calc_distance(
            self.landmark_processor.get_point(rest_landmarks, 133), 
            self.landmark_processor.get_point(rest_landmarks, 126))
        left_move = self.landmark_processor.calc_distance(
            self.landmark_processor.get_point(aligned_move_landmarks, 133), 
            self.landmark_processor.get_point(aligned_move_landmarks, 126))
        right_rest = self.landmark_processor.calc_distance(
            self.landmark_processor.get_point(rest_landmarks, 362), 
            self.landmark_processor.get_point(rest_landmarks, 355))
        right_move = self.landmark_processor.calc_distance(
            self.landmark_processor.get_point(aligned_move_landmarks, 362), 
            self.landmark_processor.get_point(aligned_move_landmarks, 355))
        
        left_l = left_rest - left_move
        right_l = right_rest - right_move
        small_l = min(left_l, right_l)
        large_l = max(left_l, right_l)
        results["sneer"] = small_l / large_l if large_l > visible_threshold else 0.0
        
        # 4. 咧嘴笑：口角横向移动
        left_rest = self.landmark_processor.get_point(rest_landmarks, 61)[0]
        left_move = self.landmark_processor.get_point(aligned_move_landmarks, 61)[0]
        right_rest = self.landmark_processor.get_point(rest_landmarks, 291)[0]
        right_move = self.landmark_processor.get_point(aligned_move_landmarks, 291)[0]
        
        left_l = abs(left_move - left_rest)
        right_l = abs(right_move - right_rest)
        small_l = min(left_l, right_l)
        large_l = max(left_l, right_l)
        results["smile"] = small_l / large_l if large_l > visible_threshold else 0.0
        
        # 5. 撅嘴：嘴角到嘴唇中线（0, 11~17）拟合直线的垂直距离
        midline_indices = [0] + list(range(11, 18))  # 嘴唇中线点索引
        def fit_plane_and_dist(landmarks, corner_idx):
            # 拟合中线点
            pts = np.array([self.landmark_processor.get_3D_point(landmarks, idx) for idx in midline_indices])
            centroid = np.mean(pts, axis=0)
            # SVD求法向量
            _, _, vh = np.linalg.svd(pts - centroid)
            normal = vh[-1]  # 平面法向量
            # 平面方程: normal[0]*(x-x0) + normal[1]*(y-y0) + normal[2]*(z-z0) = 0
            # 即: A(x-x0)+B(y-y0)+C(z-z0)=0
            # 化为: Ax+By+Cz+D=0, D=-(A*x0+B*y0+C*z0)
            A, B, C = normal
            D = -np.dot(normal, centroid)
            # 嘴角点
            p = np.array(self.landmark_processor.get_3D_point(landmarks, corner_idx))
            dist = abs(A*p[0] + B*p[1] + C*p[2] + D) / np.linalg.norm(normal)
            return dist
        left_rest = fit_plane_and_dist(rest_landmarks, 61)
        left_move = fit_plane_and_dist(aligned_move_landmarks, 61)
        right_rest = fit_plane_and_dist(rest_landmarks, 291)
        right_move = fit_plane_and_dist(aligned_move_landmarks, 291)
        print(f"Left rest: {left_rest}, Left move: {left_move}, Right rest: {right_rest}, Right move: {right_move}")
        left_l = left_rest - left_move
        right_l = right_rest - right_move
        small_l = min(left_l, right_l)
        large_l = max(left_l, right_l)
        results["pucker"] = small_l / large_l if large_l > visible_threshold else 0.0
        
        return results
    
    def add_movement_data(self, movement_ratios: Dict[str, float]):
        """添加运动幅度数据到历史记录中"""
        for expression, ratio in movement_ratios.items():
            if expression in self.movement_history:
                self.movement_history[expression].append(ratio)
    
    def clear_movement_history(self):
        """清空运动历史记录"""
        for expression in self.movement_history:
            self.movement_history[expression] = []
    
    def get_movement_summary(self, include_orientation_stats: bool = False) -> Dict[str, any]:
        """获取运动分析总结"""
        average_ratios = self.__calculate_average_movement_ratios()
        scores = self.__calculate_symmetry_scores(average_ratios)
        
        summary = {
            'average_ratios': average_ratios,
            'symmetry_scores': scores,
            'data_count': {expr: len(ratios) for expr, ratios in self.movement_history.items()}
        }
        
        if include_orientation_stats:
            # 添加朝向校正相关统计信息
            summary['orientation_correction_enabled'] = True
            summary['analysis_note'] = "分析结果已应用面部朝向校正，减少了头部朝向对运动对称性评估的影响"
        
        return summary
