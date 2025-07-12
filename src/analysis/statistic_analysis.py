import numpy as np
import cv2
from skimage import measure
from scipy.spatial.distance import euclidean

# MediaPipe鼻唇沟对应特征点索引
LEFT_NASOLABIAL_POINTS = [206, 217]   # 左侧鼻翼与嘴角附近的两个特征点
RIGHT_NASOLABIAL_POINTS = [426, 437] # 右侧鼻翼与嘴角附近的两个特征点

class statistic_analysis:
    @staticmethod
    def extract_lip_midline_diff_rank(landmarks, image, draw_result=True):
        """
        计算左右嘴角到面部中线的距离差异，平均嘴唇宽度按51mm换算。
        参数：
            landmarks: MediaPipe返回的468个人脸特征点坐标
            image: 输入人脸图像
            draw_result: 是否在图片上绘制嘴角与中线距离
        返回：
            若两侧嘴角到中线距离差超过1mm返回1，否则返回0
        """
        h, w = image.shape[:2]
        # 嘴角关键点索引
        left_lip = 61
        right_lip = 291
        # 面部中线参考点（取上下中点）
        mid_top = 10
        mid_bottom = 152
        # 3D坐标归一化
        landmarks_xyz = np.array([[landmark.x * w, landmark.y * h, landmark.z * w] for landmark in landmarks])
        # 中线点集
        midline_indices = [10,151,9,8,168,6,197,195,5,4,1,19,2,164,18,200,199,175,152]
        midline_pts = np.array([landmarks_xyz[i] for i in midline_indices])
        # 拟合平面 ax+by+cz+d=0
        # 使用SVD求解平面法向量
        centroid = np.mean(midline_pts, axis=0)
        pts_centered = midline_pts - centroid
        _, _, vh = np.linalg.svd(pts_centered)
        normal = vh[-1]
        a, b, c = normal
        d = -np.dot(normal, centroid)
        # 嘴唇宽度像素（用3D欧氏距离）
        left_lip = 61
        right_lip = 291
        lip_width_px = np.linalg.norm(landmarks_xyz[left_lip] - landmarks_xyz[right_lip])
        px_per_mm = lip_width_px / 51.0
        # 点到平面距离公式
        def point_to_plane_dist(pt, a, b, c, d):
            return abs(a*pt[0] + b*pt[1] + c*pt[2] + d) / (np.linalg.norm([a,b,c]) + 1e-8)
        left_dist_px = point_to_plane_dist(landmarks_xyz[left_lip], a, b, c, d)
        right_dist_px = point_to_plane_dist(landmarks_xyz[right_lip], a, b, c, d)
        left_mm = left_dist_px / px_per_mm
        right_mm = right_dist_px / px_per_mm
        if draw_result:
            # 可视化：投影中线点到2D图像
            for pt in midline_pts:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 2, (255,0,0), -1)
            cv2.circle(image, (int(landmarks_xyz[left_lip][0]), int(landmarks_xyz[left_lip][1])), 3, (0,255,0), -1)
            cv2.circle(image, (int(landmarks_xyz[right_lip][0]), int(landmarks_xyz[right_lip][1])), 3, (0,0,255), -1)
            cv2.putText(image, f"L:{left_mm:.2f}mm", (int(landmarks_xyz[left_lip][0]), int(landmarks_xyz[left_lip][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(image, f"R:{right_mm:.2f}mm", (int(landmarks_xyz[right_lip][0]), int(landmarks_xyz[right_lip][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        # 判断距离差
        if abs(left_mm - right_mm) > 1.0:
            return 1
        else:
            return 0
    @staticmethod
    def extract_palpebral_fissure_width_rank(landmarks, image, draw_result=True):
        """
        计算左右眼裂宽度差异，平均宽度按8.5mm换算。
        参数：
            landmarks: MediaPipe返回的468个人脸特征点坐标
            image: 输入人脸图像
            draw_result: 是否在图片上绘制眼裂宽度
        返回：
            若两侧眼裂宽度差超过1mm返回1，否则返回0
        """
        h, w = image.shape[:2]
        # 左右眼裂关键点索引（可根据实际模型微调）
        left_top, left_bottom = 159, 145
        right_top, right_bottom = 386, 374
        landmarks_xy = np.array([[landmark.x * w, landmark.y * h] for landmark in landmarks])
        left_width_px = np.linalg.norm(landmarks_xy[left_top] - landmarks_xy[left_bottom])
        right_width_px = np.linalg.norm(landmarks_xy[right_top] - landmarks_xy[right_bottom])
        # 像素到毫米换算，假定平均宽度8.5mm
        avg_px = (left_width_px + right_width_px) / 2
        px_per_mm = avg_px / 8.5
        left_mm = left_width_px / px_per_mm
        right_mm = right_width_px / px_per_mm
        if draw_result:
            cv2.line(image, tuple(landmarks_xy[left_top].astype(int)), tuple(landmarks_xy[left_bottom].astype(int)), (0,255,0), 2)
            cv2.line(image, tuple(landmarks_xy[right_top].astype(int)), tuple(landmarks_xy[right_bottom].astype(int)), (0,0,255), 2)
            cv2.putText(image, f"L:{left_mm:.2f}mm", tuple(landmarks_xy[left_top].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(image, f"R:{right_mm:.2f}mm", tuple(landmarks_xy[right_top].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        # 判断宽度差
        if abs(left_mm - right_mm) > 1.0:
            return 1
        else:
            return 0
    """
    面部统计分析类
    包含法令纹长度计算方法
    """

    @staticmethod
    def extract_nasolabial_rank(landmarks, image, draw_result=True):
        """
        参数：
            landmarks: (N, 2) 数组，MediaPipe返回的468个人脸特征点坐标
            image: 输入的人脸图像（BGR格式）
            draw_result: 是否在图片上绘制鼻唇沟
        返回：
            left_len: 左侧鼻唇沟估计长度
            right_len: 右侧鼻唇沟估计长度
            img_with_lines: 绘制了鼻唇沟的图片（如果draw_result=True）
        """
        from skimage.filters import frangi, gabor
        from skimage import morphology

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # MediaPipe 3D关键点归一化到2D像素坐标
        landmarks_xy = np.array([[landmark.x * w, landmark.y * h] for landmark in landmarks])
        left_roi_index = [128, 196, 92, 206]
        right_roi_index = [357, 419, 322, 426]

        # 构建左、右ROI多边形mask
        left_roi = np.array([landmarks_xy[i] for i in left_roi_index], dtype=np.int32)
        right_roi = np.array([landmarks_xy[i] for i in right_roi_index], dtype=np.int32)
        mask_left = np.zeros_like(gray, dtype=np.uint8)
        mask_right = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(mask_left, [left_roi], 255)
        cv2.fillPoly(mask_right, [right_roi], 255)

        def process_roi(mask, direction):
            # --- 滤波、ROI归一化 ---
            frangi_img = frangi(gray, sigmas=np.linspace(1, 3, 20))
            combined = morphology.closing(frangi_img, morphology.disk(1))
            combined_roi = np.zeros_like(combined)
            combined_roi[mask > 0] = combined[mask > 0]
            combined_roi = (combined_roi - combined_roi.min()) / (combined_roi.max() - combined_roi.min() + 1e-8)
            norm_img = (combined_roi * 255).astype(np.uint8)
            # --- 方向性膨胀 ---
            idx0, idx1 = (206, 217) if direction == 'left' else (426, 437) if direction == 'right' else (0, 1)
            vec = landmarks_xy[idx0] - landmarks_xy[idx1]
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            length = 11
            center = np.array([length//2, length//2])
            half_len = (length-1)//2
            pt1_kernel = (center - vec_norm * half_len).astype(int)
            pt2_kernel = (center + vec_norm * half_len).astype(int)
            main_kernel = np.zeros((length, length), dtype=np.uint8)
            cv2.line(main_kernel, tuple(pt1_kernel), tuple(pt2_kernel), 1, 1)
            connect = cv2.dilate(norm_img, main_kernel, iterations=1)
            # --- 自适应二值化 ---
            block_size, C = 11, -2
            roi = connect.copy(); roi[mask == 0] = 0
            local_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
            final = np.zeros_like(connect, dtype=np.uint8)
            final[mask > 0] = local_thresh[mask > 0]
            # --- 连通域筛选 ---
            labels = measure.label(final, connectivity=2)
            props = measure.regionprops(labels)
            pt0, pt1 = landmarks_xy[idx0], landmarks_xy[idx1]
            main_angle = np.arctan2((pt1-pt0)[1], (pt1-pt0)[0])
            candidate_lines = []
            vis_img = np.zeros_like(final)
            for prop in props:
                if prop.eccentricity > 0.98 and 30 < prop.area < 1000:
                    coords = np.fliplr(prop.coords)
                    if len(coords) < 2: continue
                    coords_centered = coords - np.mean(coords, axis=0)
                    comp = np.linalg.svd(coords_centered, full_matrices=False)[2][0]
                    comp_angle = np.arctan2(comp[1], comp[0])
                    angle_diff = np.abs(np.arctan2(np.sin(comp_angle-main_angle), np.cos(comp_angle-main_angle)))
                    if angle_diff > np.pi/2:
                        angle_diff = np.pi - angle_diff
                    if angle_diff < np.deg2rad(10):
                        candidate_lines.append(prop.coords)
                        for y, x in prop.coords:
                            vis_img[y, x] = 255
            return candidate_lines

        candidate_lines_left = process_roi(mask_left, direction='left')
        candidate_lines_right = process_roi(mask_right, direction='right')

        def line_near_landmarks(lines, idx0, idx1):
            pt0 = landmarks_xy[idx0]
            pt1 = landmarks_xy[idx1]
            line_vec = pt1 - pt0
            line_unit = line_vec / (np.linalg.norm(line_vec) + 1e-8)
            all_proj = []
            for coords in lines:
                coords_arr = np.array(coords)
                vecs = coords_arr - pt0
                proj_lens = np.dot(vecs, line_unit)
                all_proj.extend(proj_lens)
            if not all_proj:
                return None, None
            # 合并所有投影区间，计算总长度（去重叠）
            all_proj = np.array(all_proj)
            all_proj.sort()
            intervals = []
            step = 1.0  # 像素步长
            start = all_proj[0]
            end = all_proj[0]
            for val in all_proj[1:]:
                if val - end > step:
                    intervals.append((start, end))
                    start = val
                end = val
            intervals.append((start, end))
            length = sum(e - s for s, e in intervals)
            # 可视化用：选点数最多的连通域
            return length, lines

        # 只在各自ROI内估算左、右侧鼻唇沟长度和像素点
        left_len, left_lines = line_near_landmarks(candidate_lines_left, *LEFT_NASOLABIAL_POINTS)
        right_len, right_lines = line_near_landmarks(candidate_lines_right, *RIGHT_NASOLABIAL_POINTS)
        if left_len is None:
            left_len = 0.0
        if right_len is None:
            right_len = 0.0

        if draw_result:
            # 合并所有法令纹连通域骨架，并直接在原始image上高亮标注
            color_left = (0, 255, 0)   # 绿色
            color_right = (0, 0, 255)  # 红色
            thickness = 2
            for lines, color in zip([left_lines, right_lines], [color_left, color_right]):
                if lines is not None:
                    for coords in lines:
                        temp = np.zeros_like(gray, dtype=np.uint8)
                        for y, x in coords:
                            temp[y, x] = 1
                        skel = morphology.medial_axis(temp).astype(np.uint8)
                        # 在原图上高亮骨架像素
                        ys, xs = np.where(skel > 0)
                        for y, x in zip(ys, xs):
                            cv2.circle(image, (x, y), 0, color, thickness)
        if left_len == 0 or right_len == 0:
            return 2
        if min(left_len, right_len) / max(left_len, right_len) < 0.7:
            return 1
        else :
            return 0