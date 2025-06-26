"""
视频处理器模块
负责视频的分析、处理和输出
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from config import Config

from analysis.movement_calculator import MovementCalculator
from analysis.facial_analysis_engine import FacialAnalysisEngine


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path        
        self.movement_calculator = MovementCalculator()
        self.analysis_engine = FacialAnalysisEngine(self.movement_calculator)
        self._init_mediapipe()
        # 表情名称映射
        self.expression_keys = ['抬眉', '闭眼', '皱鼻', '咧嘴笑', '撅嘴']
    
    def _init_mediapipe(self):
        """初始化MediaPipe"""
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
    
    def process_single_video(self, video_path: str, output_dir: str) -> bool:
        """处理单个视频文件"""
        try:
            video_name = Path(video_path).stem
            video_output_dir = Path(output_dir) / video_name
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"处理视频: {video_name}")
            
            # 第一阶段：分析整个视频，找到基准帧和表情峰值帧
            baseline_frame, expression_peaks = self._analyze_video_for_peaks(video_path)
            
            if not baseline_frame:
                logging.error("未找到合适的基准帧")
                return False
            
            if not expression_peaks:
                logging.error("未找到表情峰值帧")
                return False
            
            # 第二阶段：保存基准图和表情图片，并获取标注好的帧
            annotated_frames = self._save_baseline_and_expression_images(
                video_path, baseline_frame, expression_peaks, video_output_dir, video_name)
            
            # 第三阶段：使用已标注的帧生成视频
            self._create_annotated_video(
                video_path, video_output_dir, video_name, annotated_frames)
            
            print(f"视频 {video_name} 处理完成！")
            print(f"输出目录: {video_output_dir}")
            
            summery = self.analysis_engine.movement_calculator.get_movement_summary()
            print(f"运动幅度对称性总结: {summery}")
            self.analysis_engine.movement_calculator.clear_movement_history()
            
            return True
            
        except Exception as e:
            logging.error(f"处理视频 {video_path} 时出错: {e}")
            return False
    
    def _analyze_video_for_peaks(self, video_path: str):
        """分析整个视频，找到基准帧和表情峰值帧"""
        # Pass 1: Find the baseline frame by analyzing neutral expression
        print("第一阶段-A：初步分析视频，寻找基准帧（前三秒内变化率最小）...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"无法打开视频文件 {video_path}")
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        
        first_pass_data = []
        frame_count = 0
        max_time = 3.0  # 只取前三秒
        
        with self.FaceLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(frame_count * 1000 / fps)
                timestamp_s = frame_count / fps
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                if detection_result.face_blendshapes and detection_result.face_landmarks:
                    expressions = self.analysis_engine.expression_analyzer.analyze_expressions(detection_result)
                    current_frame_info = {
                        'frame_number': frame_count,
                        'timestamp_ms': timestamp_ms,
                        'timestamp_s': timestamp_s,
                        'frame': frame.copy(),
                        'rgb_frame': rgb_frame.copy(),
                        'detection_result': detection_result,
                        'expressions': expressions
                    }
                    first_pass_data.append(current_frame_info)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"初步分析进度: {progress:.1f}% ({frame_count}/{total_frames})")
        cap.release()

        if not first_pass_data:
            logging.error("前三秒内未能找到任何有效的帧来确定基准帧。")
            return None, None

        # 计算每帧的中性状态
        neutral_values = [f['expressions']['中性状态'] for f in first_pass_data]
        timestamps = [f['timestamp_s'] for f in first_pass_data]
        # 计算变化率（绝对值）
        neutral_deltas = [0.0]
        for i in range(1, len(neutral_values)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                delta = abs((neutral_values[i] - neutral_values[i-1]) / dt)
            else:
                delta = 0.0
            neutral_deltas.append(delta)
        # 前缀和
        prefix_sum = [0.0]
        for d in neutral_deltas:
            prefix_sum.append(prefix_sum[-1] + d)
        # 滑窗平均
        window_sec = 0.1
        window_size = max(1, int(window_sec * fps))
        min_avg = float('inf')
        min_idx = 0
        for i in range(len(neutral_deltas)):
            l = max(0, i - window_size)
            r = min(len(neutral_deltas)-1, i + window_size)
            if r >= max_time * fps:
                break
            count = r - l + 1
            avg = (prefix_sum[r+1] - prefix_sum[l]) / count
            if avg < min_avg:
                min_avg = avg
                min_idx = i
        baseline_frame_info = first_pass_data[min_idx]
        print(f"基准帧已找到: 帧号 {baseline_frame_info['frame_number']} (中性值: {baseline_frame_info['expressions']['中性状态']:.3f}, 变化率窗口均值: {min_avg:.5f})")
        
        # Pass 2: Re-calculate expressions with reference landmarks and find peaks
        print("第一阶段-B：使用基准帧重新分析表情并寻找峰值...")
        reference_landmarks = baseline_frame_info['detection_result'].face_landmarks[0]
        
        final_frame_data = []
        expression_peaks = {expr: {'max_value': 0.0, 'frames': []} for expr in self.expression_keys}

        for frame_info in first_pass_data:
            reanalyzed_expressions = self.analysis_engine.expression_analyzer.analyze_expressions(
                frame_info['detection_result'], 
                reference_landmarks=reference_landmarks
            )
            
            # Update frame_info with the new expressions and clean up
            frame_info['expressions'] = reanalyzed_expressions
            frame_info['landmarks'] = frame_info['detection_result'].face_landmarks[0]
            del frame_info['detection_result']
            
            final_frame_data.append(frame_info)
            
            # Update expression peaks with the new values
            for expr_key in self.expression_keys:
                expr_value = reanalyzed_expressions[expr_key]
                if expr_value > expression_peaks[expr_key]['max_value']:
                    expression_peaks[expr_key]['max_value'] = expr_value
        
        # Find all frames that are within 90% of the peak for each expression
        print("寻找表情峰值90%的帧...")
        for expr_key in self.expression_keys:
            peak_value = expression_peaks[expr_key]['max_value']
            threshold_90 = peak_value * Config.PEAK_THRESHOLD
            
            print(f"{expr_key}: 峰值={peak_value:.3f}, 90%阈值={threshold_90:.3f}")
            
            qualifying_frames = []
            for frame_info in final_frame_data:
                if frame_info['expressions'][expr_key] >= threshold_90:
                    qualifying_frames.append(frame_info)
            
            expression_peaks[expr_key]['frames'] = qualifying_frames
            print(f"找到{len(qualifying_frames)}帧达到{expr_key}的90%峰值")
        
        return baseline_frame_info, expression_peaks
    
    def _save_baseline_and_expression_images(self, video_path, baseline_frame, expression_peaks, video_output_dir, video_name):
        """保存基准图和表情图片"""
        print("第二阶段：保存基准图和表情图片...")
        video_annotated_frames = {}
        
        # 保存基准图
        baseline_path = video_output_dir / f"{video_name}_baseline.jpg"
        print(f"保存基准图: {baseline_path}")
        
        try:
            # 使用PIL保存基准图
            if len(baseline_frame['frame'].shape) == 3:
                rgb_image = cv2.cvtColor(baseline_frame['frame'], cv2.COLOR_BGR2RGB)
            else:
                rgb_image = baseline_frame['frame']
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(str(baseline_path), 'JPEG', quality=95)
            print(f"基准图已保存: {baseline_path}")
        except Exception as e:
            print(f"基准图保存失败: {e}")
            logging.error(f"无法保存基准图: {baseline_path}, 错误: {e}")
        
        print(f"基准图中性值: {baseline_frame['expressions']['中性状态']:.3e}")
        
        # 为每个表情创建目录并保存图片
        for expr_key in self.expression_keys:
            expr_frames = expression_peaks[expr_key]['frames']
            if not expr_frames:
                print(f"警告: {expr_key} 没有找到合适的帧")
                continue
            
            expr_dir = video_output_dir / f"{expr_key}_images"
            expr_dir.mkdir(exist_ok=True)
            
            print(f"保存 {expr_key} 的 {len(expr_frames)} 张图片...")
            
            for i, frame_info in enumerate(expr_frames):
                # 计算运动幅度对称性
                movement_ratios = self.analysis_engine.movement_calculator.calculate_facial_movement_ratios(
                    baseline_frame['landmarks'], frame_info['landmarks'])
                
                # 处理帧以添加标注
                # 创建模拟的detection_result对象
                class MockBlendshape:
                    def __init__(self, category_name, score):
                        self.category_name = category_name
                        self.score = score

                mock_blendshapes = []
                for k, v in self._convert_expressions_to_blendshapes(frame_info['expressions']).items():
                    mock_blendshapes.append(MockBlendshape(k, v))

                class MockDetectionResult:
                    def __init__(self, landmarks, blendshapes):
                        self.face_landmarks = [landmarks]
                        self.face_blendshapes = [blendshapes]
                
                mock_detection_result = MockDetectionResult(frame_info['landmarks'], mock_blendshapes)
                
                # 使用特定表情处理方法，只标注当前表情
                annotated_frame, _ = self.analysis_engine.process_frame_for_specific_expression(
                    frame_info['rgb_frame'], 
                    mock_detection_result,
                    expr_key,  # 传入当前表情名称
                    baseline_frame['landmarks'])
                
                # 为视频准备已标注的帧
                frame_number = frame_info['frame_number']
                current_value = frame_info['expressions'][expr_key]

                if frame_number not in video_annotated_frames or current_value > video_annotated_frames[frame_number]['value']:
                    video_annotated_frames[frame_number] = {
                        'frame': annotated_frame,
                        'value': current_value
                    }
                
                # 保存标注图片
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                img_path = expr_dir / f"{video_name}_{expr_key}_{i+1:03d}_frame{frame_info['frame_number']:06d}.jpg"
                
                # 使用PIL保存图片
                try:
                    rgb_image = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    pil_image.save(str(img_path), 'JPEG', quality=95)
                except Exception as e:
                    print(f"保存图片失败: {e}")
                    logging.error(f"保存图片失败: {img_path}, 错误: {e}")
                    continue
                
                # 保存图片信息
                img_info = {
                    'frame_number': frame_info['frame_number'],
                    'timestamp_ms': frame_info['timestamp_ms'],
                    'expression': expr_key,
                    'expression_value': frame_info['expressions'][expr_key],
                    'expressions': frame_info['expressions'],
                    'movement_ratios': movement_ratios
                }
                
                info_path = expr_dir / f"{video_name}_{expr_key}_{i+1:03d}_frame{frame_info['frame_number']:06d}_info.json"
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(img_info, f, indent=2, ensure_ascii=False)
            
            print(f"{expr_key} 图片保存完成，共 {len(expr_frames)} 张")

        final_video_frames = {fn: data['frame'] for fn, data in video_annotated_frames.items()}
        return final_video_frames
    
    def _convert_expressions_to_blendshapes(self, expressions):
        """将表情字典转换为blendshape格式"""
        # 这是一个简化的映射，实际应用中可能需要更复杂的映射
        blendshape_mapping = {
            '抬眉': 'browInnerUp',
            '闭眼': 'eyeBlinkLeft', 
            '皱鼻': 'noseSneerLeft',
            '咧嘴笑': 'mouthSmileLeft',
            '撅嘴': 'mouthPucker',
            '中性状态': 'neutral'
        }
        
        blendshapes = {}
        for expr_name, value in expressions.items():
            if expr_name in blendshape_mapping:
                blendshapes[blendshape_mapping[expr_name]] = value
        
        return blendshapes
    
    def _create_annotated_video(self, video_path, video_output_dir, video_name, annotated_frames):
        """创建标注视频（使用已标注好的帧）"""
        print("第三阶段：生成标注视频...")
        
        annotated_frame_numbers = set(annotated_frames.keys())
        print(f"需要标注的帧数: {len(annotated_frame_numbers)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"无法打开视频文件 {video_path}")
            return
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = video_output_dir / f"{video_name}_annotated.mp4"
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 如果当前帧需要标注
            if frame_count in annotated_frame_numbers:
                annotated_frame = annotated_frames[frame_count]
                # 转换为BGR并写入视频
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(annotated_frame_bgr)
            else:
                # 不需要标注的帧，直接写入原始帧
                video_writer.write(frame)
            
            frame_count += 1
            
            # 显示进度
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"视频处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        video_writer.release()
        
        print(f"标注视频已保存: {output_video_path}")
        print(f"共标注了 {len(annotated_frame_numbers)} 帧")
