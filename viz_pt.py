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
    
    for key in ['idx', 'part_axis', 'part_pv_point']:
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
    print(pivot_np)
    for i in range(3):
        if pivot_np[0] == 0:
            pivot_np[0] = 1e-15
    print(pivot_np)
    
    joint_rotation = drct_rotation(torch.tensor(pivot_np, dtype=torch.float32).reshape(1,1,3)).reshape(3,3)
    j = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0075, cone_radius=0.015, cylinder_height=0.3, cone_height=0.075)
    j.rotate([[1,0,0],[0,1,0],[0,0,-1]], center=False)
    j.rotate(joint_rotation, center=False)
    j.translate(direction_np.astype(np.float64))
    j.paint_uniform_color([0,0,0])
    
    num_labels = len(np.unique(label_np))  # ラベルの種類数
    unique_labels = np.unique(label_np)
    #print(num_labels, unique_labels)
    label_np -= min(unique_labels)
    unique_labels -= min(unique_labels)
    #print(label_np)

    # ラベルごとに色を設定（カラーマップを使用）
    colormap = plt.get_cmap("tab10")  # カラーマップを変更してコントラストを強調
    #label_colors = colormap(unique_labels / (num_labels - 1))[:, :3]  # RGB値を取得
    label_colors = colormap(np.arange(num_labels) / (num_labels - 1))[:, :3]  # RGB値を取得

    # ラベルに対応する色を割り当て
    rgb_colors = np.array([label_colors[np.where(unique_labels == label)[0][0]] for label in label_np])

    # Open3D PointCloud オブジェクトを作成
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_np)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb_colors)
    
    
    # 点群を可視化
    o3d.visualization.draw_geometries([point_cloud, j], window_name=file_path)


def viz_syn_result():
    viz = np.load('log/safe_test/model_20241223_143106/viz/00000.npz', allow_pickle=True)['arr_0'].item()
    print(viz.keys())
    
    
def eval_result(csv_path):
    eval_csv = pd.read_csv(csv_path)
    print(np.mean(eval_csv['seg_1']))

if __name__ == "__main__":
    
    file_paths = glob.glob("/home/akrobo/research/op_align/dataset/pc/partial/safe/test-50/*")
    #for file_path in file_paths:
    #    viz_pt(file_path)
    
    viz_pt("dataset/pc/partial/safe/test-100/101594_100p_joint_0.pt")
    #eval_result("log/safe-50_test/model_20241231_123155/csv/safe-50_eval.csv")