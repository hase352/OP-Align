import torch
import numpy as np
import open3d as o3d
import os
import re



"""
sapienでlabelを取ってくるが、label==2のlink がbaseにくっついている
"""
PARTIAL_ROOT_PATH = "/home/akrobo/research/op_align/real/pc/partial"

def create_syn_data(points, labels, data_info: str, direction, pivot, per_object=False, hsaur_itr_num=-1):
    data = torch.load(os.path.join(PARTIAL_ROOT_PATH,"safe/0.pt"))#必要ないデータを埋めるため
    shape_id = int(data_info.split("_")[0])
    open_percentage = int(re.findall(r'\d+',  data_info.split("_")[1])[0])
    file_idx = int(f"{shape_id}{open_percentage:03}")#下三桁がpercentageで、その前がshape_id

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 3. ダウンサンプリング
    voxel_size = 0.06  # ボクセルサイズ
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    # 4. ダウンサンプリング後の点群をNumPy配列に変換
    downsampled_points = np.asarray(downsampled_point_cloud.points)
    print("ダウンサンプル後",downsampled_points.shape)
    
    indices = []
    for point in downsampled_points:
        id = np.argmin(np.linalg.norm(points - point, axis=1))
        indices.append(id)
    indices = np.array(indices)

    # ダウンサンプリング後のラベルを取得
    down_labels = labels[indices]
        
    # 5. ランダムサンプリングして1024点を取得
    if downsampled_points.shape[0] >= 1024:
        sampled_indices = np.random.choice(downsampled_points.shape[0], 1024, replace=False)
        sampled_points = downsampled_points[sampled_indices]
        sampled_labels = down_labels[sampled_indices]
    else:
        # 点が不足する場合は補間する（簡易的な補間例）
        repeat_factor = int(np.ceil(1024 / downsampled_points.shape[0]))
        extended_points = np.tile(downsampled_points, (repeat_factor, 1))
        sampled_points = extended_points[:1024]
        extended_labels = np.tile(down_labels, (repeat_factor, 1))
        sampled_labels = extended_labels[:1024]
        
    print("サンプル後",sampled_points.shape)
    #point_cloud = o3d.geometry.PointCloud()
    #point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
    #o3d.visualization.draw_geometries([point_cloud], window_name="1024")
    
    sampled_points = sampled_points.astype(np.float32)
    sampled_labels = sampled_labels.astype(np.int64)
    sampled_labels -= min(sampled_labels)
    print(np.unique(sampled_labels))
    for i in range(sampled_labels.shape[0]):
        if sampled_labels[i] != 0:
            sampled_labels[i] = 1
            
    print(direction, pivot)
    data['pc'] = torch.from_numpy(sampled_points)
    data['idx'] = torch.tensor([file_idx], dtype=torch.int64)
    print(torch.tensor([file_idx], dtype=torch.int64))
    data['label'] = torch.from_numpy(sampled_labels)
    data['part_axis'] = torch.from_numpy(np.array([pivot.astype(np.float32)]))
    data['part_pv_point'] = torch.from_numpy(np.array([direction.astype(np.float32)]))
    print([torch.from_numpy(pivot.astype(np.float32))])
    if hsaur_itr_num != -1:
        if not os.path.exists(os.path.join(PARTIAL_ROOT_PATH, "safe-hsaur", str(shape_id))):
            os.mkdir(os.path.join(PARTIAL_ROOT_PATH, "safe-hsaur", str(shape_id)))
        if not os.path.exists(os.path.join(PARTIAL_ROOT_PATH, "safe-hsaur", str(shape_id), "test-" + str(open_percentage))):
            os.mkdir(os.path.join(PARTIAL_ROOT_PATH, "safe-hsaur", str(shape_id), "test-" + str(open_percentage)))
        torch.save(data, os.path.join(PARTIAL_ROOT_PATH, "safe-hsaur", str(shape_id), "test-" + str(open_percentage), str(hsaur_itr_num) + ".pt"))
        print("Save testdata: ", str(hsaur_itr_num) + ".pt  to  safe-hsaur/", str(shape_id), "/test-", str(open_percentage))
    else:
        if per_object == False:
            if not os.path.exists(os.path.join(PARTIAL_ROOT_PATH, "safe", "test-" + str(open_percentage))):
                os.mkdir(os.path.join(PARTIAL_ROOT_PATH, "safe", "test-" + str(open_percentage)))
            torch.save(data, os.path.join(PARTIAL_ROOT_PATH, "safe", "test-" + str(open_percentage) , data_info + ".pt"))
            print("Save testdata: ", data_info + ".pt  to  safe/test-"+str(open_percentage))
        else:
            if not os.path.exists(os.path.join(PARTIAL_ROOT_PATH, "safe-object", str(shape_id))):
                os.mkdir(os.path.join(PARTIAL_ROOT_PATH, "safe-object", str(shape_id)))
            torch.save(data, os.path.join(PARTIAL_ROOT_PATH, "safe-object", str(shape_id), data_info + ".pt"))
            print("Save testdata: ", data_info + ".pt  to  safe-object/"+str(shape_id))
    
def check():
    data = torch.load(PARTIAL_ROOT_PATH+"/safe/test/0.pt")#必要ないデータを埋めるため
    created_data = torch.load("dataset/pc/partial/safe/test1/101363_10p_joint_0.pt")
    
    print("data",data['pc'].dtype, data['label'].dtype)
    print("created_data",created_data['pc'].dtype, created_data['label'].dtype)
    
if __name__ == "__main__":
    #create_syn_data([[0,0,0]], 0, "1423_50p_joint_0")
    check()