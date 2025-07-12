"""
联动运动幅度计算器
"""

from typing import Dict, List, Tuple
from core.landmark_processor import LandmarkProcessor
import numpy as np
from config import Config
from collections import defaultdict


class SynkinesisCalculator:
    """运动幅度计算器"""
    def __calculate_average_movement_ratios(self) -> Dict[str, float]:
        """计算各表情联带运动幅度的平均值"""
        averages = {}
        for expression, ratios in self.movement_history.items():
            if ratios:
                averages[expression] = sum(ratios) / len(ratios)
            else:
                averages[expression] = 0.0
        return averages
    
    def __calculate_symmetry_scores(self, average_ratios: Dict[str, float] = None) -> Dict[str, int]:
        """
        根据联带运动幅度比率计算评分
        评分标准：
        0到3分
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
                if ratio <= 0.5:  # 几乎不动
                    score = 0
                elif ratio <= 1.5:
                    score = 1
                elif ratio <= 2.5:
                    score = 2
                else:  # 91%以上
                    score = 3
                
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
    
    def calculate_facial_movement(self, rest_blendshape, move_blendshape, expression: str) -> float:
        # 将blendshapes转换为字典便于查找
        blendshape_dict = {}
        blendshape_diff_dict = {}
        category_list = []
        for blendshape in rest_blendshape:
            blendshape_dict[blendshape.category_name] = float(blendshape.score)
        for blendshape in move_blendshape:
            if blendshape.category_name in blendshape_dict:
                diff = float(blendshape.score) - blendshape_dict[blendshape.category_name]
                blendshape_dict[blendshape.category_name] = diff / blendshape_dict[blendshape.category_name]
        blendshape_diff_dict = defaultdict(float)
        for blendshape in move_blendshape:
            base_name = blendshape.category_name.replace('Left', '').replace('Right', '')
            if 'Left' in blendshape.category_name:
                blendshape_diff_dict[base_name] += float(blendshape.score)
            elif 'Right' in blendshape.category_name:
                blendshape_diff_dict[base_name] -= float(blendshape.score)
            if base_name not in category_list:
                category_list.append(base_name)

        # 面部区域关键词映射
        regions = ['brow', 'eye', 'cheek', 'nose', 'mouth', 'jaw']
        region_grades = {}
        region_len = {region: 0 for region in regions}
        for cat in category_list:
            blendshape_diff_dict[cat] = abs(blendshape_diff_dict[cat])
            if blendshape_diff_dict[cat] <= 0.1:
                blendshape_diff_dict[cat] = 0.0
            else :
                for region in regions:
                    if region in cat:
                        region_len[region] += 1 
        for region in regions:
            # 筛选包含该区域关键词的类别
            region_categories = [cat for cat in category_list if region in cat]
            # 计算该区域的平均分数
            if region_categories and region_len[region] > 0:
                region_grades[f'{region}_grade'] = sum(blendshape_diff_dict[cat] for cat in region_categories) / region_len[region]
            else:
                region_grades[f'{region}_grade'] = 0.0
        result = 0
        if expression == '抬眉':
            grade = region_grades['eye_grade']
            if grade > 0.2:
                result = 1
            grade = region_grades['cheek_grade']
            grade += region_grades['nose_grade']
            if grade > 0.3: 
                result = 2
            grade = region_grades['mouth_grade']
            grade += region_grades['jaw_grade']
            if grade > 0.3: 
                result = 3
        if expression == '闭眼':
            grade = region_grades['brow_grade']
            grade += region_grades['cheek_grade']
            if grade > 0.2:
                result = 1
            grade = region_grades['nose_grade']
            if grade > 0.2:
                result = 2
            grade = region_grades['jaw_grade']
            grade += region_grades['mouth_grade']
            if grade > 0.3:
                result = 3
        if expression == '皱鼻':
            grade = region_grades['brow_grade']
            grade += region_grades['cheek_grade']
            if grade > 0.4:
                result = 1
            grade = region_grades['eye_grade']
            if grade > 0.1:
                result = 2
            grade = region_grades['jaw_grade']
            grade += region_grades['mouth_grade'] * 3
            if grade > 0.4:
                if result == 2: 
                    result = 3
                else:
                    result = 2
        if expression == '咧嘴笑':
            grade = region_grades['nose_grade']
            if grade > 0.2:
                result = 1
            grade = region_grades['eye_grade']
            if grade > 0.1:
                result = 2
            grade = region_grades['brow_grade']
            if grade > 0.1:
                result = 3
        if expression == '撅嘴':
            grade = region_grades['nose_grade']
            if grade > 0.2:
                result = 1
            grade = region_grades['eye_grade']
            if grade > 0.1:
                result = 2
            grade = region_grades['brow_grade']
            if grade > 0.1:
                result = 3
        return result

    def add_movement_data(self, movement_ratios, expression):
        """添加联带运动幅度数据到历史记录中"""
        expression_mapping = {
            "抬眉": "raise_eyebrow",
            "闭眼": "blink",
            "皱鼻": "sneer",
            "咧嘴笑": "smile",
            "撅嘴": "pucker"
        }
        self.movement_history[expression_mapping[expression]].append(movement_ratios)

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
        
        return summary
