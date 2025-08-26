import os
import numpy as np
from tqdm import tqdm

# 配置路径（根据你目录修改）
KITTI_ROOT = 'data/kitti'
LABEL_DIR = os.path.join(KITTI_ROOT, 'training', 'label_2')
VELO_DIR = os.path.join(KITTI_ROOT, 'training', 'velodyne')

# 小目标筛选参数
MAX_L, MAX_W, MAX_H = 2.5, 1.0, 1.5
MAX_POINTS = 10

# KITTI 类别
VALID_CLASSES = ['Car', 'Pedestrian', 'Cyclist']

def read_velodyne(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # x, y, z

def inside_box(points, center, size, ry):
    '''判断哪些点在 3D 边界框内（粗略 AABB 判断）'''
    l, w, h = size
    x, y, z = points.T
    cx, cy, cz = center

    in_x = np.logical_and(x > cx - l/2, x < cx + l/2)
    in_y = np.logical_and(y > cy - h/2, y < cy + h/2)
    in_z = np.logical_and(z > cz - w/2, z < cz + w/2)

    return np.where(in_x & in_y & in_z)[0]

def process_frame(frame_id):
    label_path = os.path.join(LABEL_DIR, f'{frame_id}.txt')
    velo_path = os.path.join(VELO_DIR, f'{frame_id}.bin')

    if not os.path.exists(label_path) or not os.path.exists(velo_path):
        return False

    points = read_velodyne(velo_path)

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls = parts[0]
        if cls not in VALID_CLASSES:
            continue

        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        if l > MAX_L or w > MAX_W or h > MAX_H:
            continue

        x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
        ry = float(parts[14])

        indices = inside_box(points, center=(x, y, z), size=(l, w, h), ry=ry)

        if len(indices) < MAX_POINTS:
            return True  # 该帧存在一个小且稀疏目标

    return False

# 扫描所有 training 样本
all_ids = sorted([f.replace('.txt', '') for f in os.listdir(LABEL_DIR)])
selected_ids = []

print("正在筛选小且稀疏目标帧...")
for id in tqdm(all_ids):
    if process_frame(id):
        selected_ids.append(id)

# 输出结果
save_path = 'small_sparse_frames.txt'
with open(save_path, 'w') as f:
    for id in selected_ids:
        f.write(id + '\n')

print(f"\n共找到 {len(selected_ids)} 个包含小目标且点云点数少于10的帧。")
print(f"结果已保存到 {save_path}")
