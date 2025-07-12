"""
表情信息可视化器
用于在图像上绘制表情分析结果
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from PIL import Image, ImageDraw

from .visualizers import BaseVisualizer


class ExpressionVisualizer(BaseVisualizer):
    """表情信息可视化器"""
    
    def draw(self, image: np.ndarray, expressions: Dict[str, float], 
             target_expression: str, movement_ratios: Optional[Dict[str, float]] = None, synkinesis_scores: Optional[float] = None) -> np.ndarray:
        """在图像上只绘制指定的表情信息和对应的运动幅度比率"""
        h, w = image.shape[:2]
        
        # 转换为PIL图像以支持中文显示
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        font = self.font_manager.get_chinese_font(20)
        small_font = self.font_manager.get_chinese_font(16)
        
        # 计算背景高度 - 只显示目标表情和中性状态，以及对应的运动比率
        items_to_show = 2  # 目标表情 + 中性状态
        if movement_ratios:
            items_to_show += 2  # 标题 + 对应的运动比率
        if synkinesis_scores:
            items_to_show += 2  # 标题 + 联动运动幅度
        box_height = 30 * items_to_show + 20
        
        # 创建背景矩形
        background = Image.new('RGBA', (380, box_height), (0, 0, 0, 180))
        pil_image.paste(background, (10, 10), background)
        
        # 绘制表情信息标题
        y_offset = 20
        draw.text((20, y_offset), "表情分析:", font=font, fill=(255, 255, 255))
        y_offset += 25
        
        # 只显示目标表情和中性状态
        expressions_to_show = [target_expression, '中性状态']
        for expr_name in expressions_to_show:
            if expr_name in expressions:
                value = expressions[expr_name]
                color = self.__get_expression_color(expr_name, value)
                text = self.__format_expression_text(expr_name, value)
                
                draw.text((30, y_offset), text, font=small_font, fill=color)
                
                # 绘制激活强度条
                bar_width = int(150 * value)
                if bar_width > 0:
                    draw.rectangle([(220, y_offset + 2), (220 + bar_width, y_offset + 12)], fill=color)
                draw.rectangle([(220, y_offset + 2), (370, y_offset + 12)], outline=color, width=1)
                
                y_offset += 25
        
        # 绘制对应的运动幅度比率
        if movement_ratios:
            y_offset = self.__draw_specific_movement_ratio(draw, movement_ratios, target_expression, y_offset, font, small_font)
            y_offset += 5  # 添加额外的间距
        # 绘制联动运动幅度分数（0~3分，用条形图和数值）
        if synkinesis_scores is not None:
            y_offset = self.__draw_synkinesis_score(draw, synkinesis_scores, y_offset, font, small_font)
            y_offset += 5

        # 转换回OpenCV格式
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_image

    def __draw_synkinesis_score(self, draw, synkinesis_scores, y_offset, font, small_font):
        """绘制联动运动幅度分数（0~3分）"""
        draw.text((20, y_offset), "联动运动幅度:", font=font, fill=(255, 255, 255))
        y_offset += 25
        score = float(synkinesis_scores)
        # 颜色：0分灰色，1分绿色，2分黄色，3分红色
        if score >= 3:
            color = (255, 0, 0)      # 红色
        elif score >= 2:
            color = (255, 255, 0)    # 黄色
        elif score >= 1:
            color = (0, 255, 0)      # 绿色
        else:
            color = (180, 180, 180)  # 灰色
        # 绘制分数文本
        draw.text((30, y_offset), f"分数: {score:.1f} / 3", font=small_font, fill=color)
        # 绘制分数条
        bar_width = int(50 * score)  # 0~3分，最大150像素
        if bar_width > 0:
            draw.rectangle([(120, y_offset + 2), (120 + bar_width, y_offset + 12)], fill=color)
        draw.rectangle([(120, y_offset + 2), (120 + 150, y_offset + 12)], outline=color, width=1)
        y_offset += 25
        return y_offset
    
    def __get_expression_color(self, expr_name: str, value: float) -> Tuple[int, int, int]:
        """获取表情的显示颜色"""
        if expr_name == '中性状态':
            if value > 0.8:
                return (0, 255, 0)    # 绿色 - 很中性
            elif value > 0.6:
                return (127, 255, 0)  # 黄绿色 - 较中性
            elif value > 0.4:
                return (255, 255, 0)  # 黄色 - 中等中性
            elif value > 0.2:
                return (255, 165, 0)  # 橙色 - 轻微中性
            else:
                return (255, 0, 0)    # 红色 - 很不中性
        else:
            if value > 0.7:
                return (255, 0, 0)    # 红色 - 强烈
            elif value > 0.4:
                return (255, 165, 0)  # 橙色 - 中等
            elif value > 0.2:
                return (255, 255, 0)  # 黄色 - 轻微
            else:
                return (255, 255, 255)  # 白色 - 很弱
    
    def __format_expression_text(self, expr_name: str, value: float) -> str:
        """格式化表情文本"""
        if expr_name == '中性状态':
            return f"{expr_name}: {value:.3e} ★"  # 科学计数法，保留三位有效数字
        else:
            return f"{expr_name}: {value:.3f}"
    
    def __draw_specific_movement_ratio(self, draw, movement_ratios, target_expression, y_offset, font, small_font):
        """绘制特定表情对应的运动幅度比率"""
        # 表情与运动比率的映射
        expression_to_ratio = {
            '抬眉': 'raise_eyebrow',
            '闭眼': 'blink',
            '皱鼻': 'sneer',  
            '咧嘴笑': 'smile',
            '撅嘴': 'pucker'
        }
        
        ratio_names = {
            'raise_eyebrow': '抬眉对称性',
            'blink': '闭眼对称性',
            'sneer': '皱鼻对称性',
            'smile': '咧嘴对称性', 
            'pucker': '撅嘴对称性'
        }
        
        if target_expression in expression_to_ratio:
            ratio_key = expression_to_ratio[target_expression]
            if ratio_key in movement_ratios:
                y_offset += 5
                draw.text((20, y_offset), "运动对称性:", font=font, fill=(255, 255, 255))
                y_offset += 25
                
                value = movement_ratios[ratio_key]
                ratio_name = ratio_names.get(ratio_key, ratio_key)
                
                # 根据比率选择颜色
                if value > 0.8:
                    color = (0, 255, 0)    # 绿色 - 很对称
                elif value > 0.6:
                    color = (255, 255, 0)  # 黄色 - 较对称
                elif value > 0.3:
                    color = (255, 165, 0)  # 橙色 - 不太对称
                else:
                    color = (255, 0, 0)    # 红色 - 很不对称
                
                text = f"{ratio_name}: {value:.3f}"
                draw.text((30, y_offset), text, font=small_font, fill=color)
                
                # 绘制比率条
                bar_width = int(150 * value)
                if bar_width > 0:
                    draw.rectangle([(220, y_offset + 2), (220 + bar_width, y_offset + 12)], fill=color)
                draw.rectangle([(220, y_offset + 2), (370, y_offset + 12)], outline=color, width=1)
                
                y_offset += 25
        
        return y_offset
