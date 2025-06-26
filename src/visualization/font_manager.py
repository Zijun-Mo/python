"""
字体管理器
用于管理中文字体的加载和缓存
"""

import os
import logging
from PIL import ImageFont


class FontManager:
    """字体管理器"""
    
    def __init__(self):
        self._font_cache = {}
    
    def get_chinese_font(self, size: int = 16):
        """获取中文字体"""
        if size in self._font_cache:
            return self._font_cache[size]
        
        try:
            # Windows系统中文字体路径
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
                "C:/Windows/Fonts/simsun.ttc"   # 宋体
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, size)
                    self._font_cache[size] = font
                    return font
            
            # 如果找不到中文字体，使用默认字体
            font = ImageFont.load_default()
            self._font_cache[size] = font
            return font
            
        except Exception as e:
            logging.warning(f"字体加载失败: {e}")
            font = ImageFont.load_default()
            self._font_cache[size] = font
            return font
