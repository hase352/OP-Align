import torch
import numpy as np
import open3d as o3d
import os
import re



"""
sapienでlabelを取ってくるが、label==2のlink がbaseにくっついている
"""
PARTIAL_ROOT_PATH = "/home/akrobo/research/op_align/real/pc/partial"

def create_syn_data(points, labels, data_info: str, direction, pivot, per_object=False, hsaur_itr_num=-1):#hsaur_it_num!=-1 のならhsaur-opalignの結果
    data = torch.load(os.path.join(PARTIAL_ROOT_PATH,"safe/0.pt"))#必要ないデータを埋めるため
    shape_id = int(data_info.split("_")[0])
    open_percentage = int(re.findall(r'\d+',  data_info.split("_")[1])[0])
    if hsaur_itr_num == -1:
        file_idx = int(f"{shape_id}{open_percentage:03}")#下三桁がpercentageで、その前がshape_id
    else:
        file_idx = int(f"{shape_id}{hsaur_itr_num:03}")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 3. ダウンサンプリング
    voxel_size = 0.06  # ボクセルサイズ
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    # 4. ダウンサンプリング後の点群をNumPy配列に変換
    downsampled_points = np.asarray(downsampled_point_cloud.points)
    #print("ダウンサンプル後",downsampled_points.shape)
    
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
        extended_labels = np.tile(down_labels, repeat_factor)
        sampled_labels = extended_labels[:1024]
        
    #print("サンプル後",sampled_points.shape)
    #point_cloud = o3d.geometry.PointCloud()
    #point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
    #o3d.visualization.draw_geometries([point_cloud], window_name="1024")
    
    sampled_points = sampled_points.astype(np.float32)
    sampled_labels = sampled_labels.astype(np.int64)
    sampled_labels -= np.min(sampled_labels)
    #print(np.unique(sampled_labels))
    mean_z_of_movable_link = 0
    for i in range(sampled_labels.shape[0]):
        if sampled_labels[i] != 0:
            sampled_labels[i] = 1
            mean_z_of_movable_link += sampled_points[i, 2]
    
    #jointのdirectionのz座標がうまく計算できずのでおかしいので無理やり直す
    mean_z_of_movable_link /= np.count_nonzero(sampled_labels == 1)
    direction[2] = mean_z_of_movable_link
            
    #print(direction, pivot)
    data['pc'] = torch.from_numpy(sampled_points)
    data['idx'] = torch.tensor([file_idx], dtype=torch.int64)
    #print(torch.tensor([file_idx], dtype=torch.int64))
    data['label'] = torch.from_numpy(sampled_labels)
    data['part_axis'] = torch.from_numpy(np.array([pivot.astype(np.float32)]))
    data['part_pv_point'] = torch.from_numpy(np.array([direction.astype(np.float32)]))
    #print([torch.from_numpy(pivot.astype(np.float32))])
    
    # 全ての点群を追加してみる
    data['full_pc'] = torch.from_numpy(points.astype(np.float32))
    data['full_label'] = torch.from_numpy(labels.astype(np.int64))
    
    if hsaur_itr_num == -1:
        if per_object and str(shape_id) in ["101564", "101593", "101599",  "101604", "101605", "101611", 
                                            "101612", "101613", "101619", "101623", "102301", "102316", 
                                            "102318", "102380", "102381", "102423"]:#train data
            root_path = os.path.join(PARTIAL_ROOT_PATH, "safe-obj-train", str(shape_id))
            if not os.path.exists(root_path):
                os.mkdir(root_path)
            torch.save(data, os.path.join(root_path, data_info + ".pt"))
            print("Save testdata: ", data_info + ".pt  to  safe-obj-train/"+str(shape_id))
            
        elif per_object and str(shape_id) in ["101363", "101579", "101584", "101591", "101594", "101603", "102278",
                                              "102309", "102311", "102384", "102387", "102389", "102418"]:
            root_path = os.path.join(PARTIAL_ROOT_PATH, "safe-obj-test", str(shape_id))
            if not os.path.exists(root_path):
                os.mkdir(root_path)
            torch.save(data, os.path.join(root_path, data_info + ".pt"))
            print("Save testdata: ", data_info + ".pt  to  safe-obj-test/"+str(shape_id))
            
        else:
            dir_path = os.path.join(PARTIAL_ROOT_PATH, "safe-perct", "4", "test-" + str(open_percentage))
            os.makedirs(dir_path, exist_ok=True)
            torch.save(data, os.path.join(dir_path , data_info + ".pt"))
            print("Save testdata: ", data_info + ".pt  to  " + dir_path)
            
    else:
        dir_path = os.path.join(PARTIAL_ROOT_PATH, "safe-ours", "1", str(shape_id), "test-" + str(open_percentage))
        os.makedirs(dir_path, exist_ok=True)
        torch.save(data, os.path.join(dir_path, str(hsaur_itr_num) + ".pt"))
        print("Save testdata: ", str(hsaur_itr_num) + ".pt  to  " + dir_path)
        
            
    
def check():
    data = torch.load(PARTIAL_ROOT_PATH+"/safe/test/0.pt")#必要ないデータを埋めるため
    created_data = torch.load("dataset/pc/partial/safe/test1/101363_10p_joint_0.pt")
    
    print("data",data['pc'].dtype, data['label'].dtype)
    print("created_data",created_data['pc'].dtype, created_data['label'].dtype)
    
if __name__ == "__main__":
    #create_syn_data([[0,0,0]], 0, "1423_50p_joint_0")
    check()