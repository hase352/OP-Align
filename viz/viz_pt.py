import torch
import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
from viz import hat, ExpSO3, drct_rotation


def viz_pt(file_path):
    data = torch.load(file_path)  # ファイル名は適宜変更
    
    for key in data.keys():
        print(key, ": ",data[key].shape, data[key].dtype , data[key])
    #print(data['idx'])
    pc = data['pc']  # 'pc'キーの点群データを取得
    label = data['label']
    direction = data['part_pv_point'][0]#関節の中心
    pivot = data['part_axis'][0]#関節の軸
    #print(max(pc[:,0]), min(pc[:,0]),max(pc[:,1]), min(pc[:,1]), max(pc[:,2]), min(pc[:,2]))

    # テンソルを NumPy 配列に変換
    pc_np = pc.numpy()
    label_np = label.numpy()
    direction_np = direction.numpy()
    pivot_np = pivot.numpy()
    # print(pivot_np)
    for i in range(3):
        if pivot_np[0] == 0:
            pivot_np[0] = 1e-15
    # print(pivot_np)
    
    joint_rotation = drct_rotation(torch.tensor(pivot_np, dtype=torch.float32).reshape(1,1,3)).reshape(3,3)
    j = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0075, cone_radius=0.015, cylinder_height=0.3, cone_height=0.075)
    j.rotate([[1,0,0],[0,1,0],[0,0,-1]], center=False)
    j.rotate(joint_rotation, center=False)
    j.translate(direction_np.astype(np.float64))
    j.paint_uniform_color([0,0,0])
    
   

    rgb_colors = np.array([[0, 0, 1] if label == 0 else [1, 0, 0] for label in label_np])

    # Open3D PointCloud オブジェクトを作成
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_np)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb_colors)
    
    
    # 点群を可視化
    o3d.visualization.draw_geometries([point_cloud, j], window_name=file_path)

    

if __name__ == "__main__":
    file_paths = glob.glob("/home/hasegawa/research/efficient_manip/OP_Align/real/pc/partial/safe-ours/data_20250707_191122/101591/test-30/4.pt")
    for file_path in file_paths:
        viz_pt(file_path)