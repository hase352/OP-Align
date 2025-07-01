import torch
import numpy as np
import open3d as o3d

from glob import glob
import os
import cv2

def hat(phi):
    phi_x = phi[..., 0]
    phi_y = phi[..., 1]
    phi_z = phi[..., 2]
    zeros = torch.zeros_like(phi_x)

    phi_hat = torch.stack([
        torch.stack([zeros, -phi_z,  phi_y], dim=-1),
        torch.stack([phi_z,  zeros, -phi_x], dim=-1),
        torch.stack([-phi_y,  phi_x,  zeros], dim=-1)
    ], dim=-2)
    return phi_hat

def ExpSO3(phi, eps=1e-4):
    theta = torch.norm(phi, dim=-1)
    phi_hat = hat(phi)
    E = torch.eye(3, device=phi.device)
    coef1 = torch.zeros_like(theta)
    coef2 = torch.zeros_like(theta)

    ind = theta < eps

    # strict
    _theta = theta[~ind]
    coef1[~ind] = torch.sin(_theta) / _theta
    coef2[~ind] = (1 - torch.cos(_theta)) / _theta**2

    # approximate
    _theta = theta[ind]
    _theta2 = _theta**2
    _theta4 = _theta**4
    coef1[ind] = 1 - _theta2/6 + _theta4/120
    coef2[ind] = .5 - _theta2/24 + _theta4/720

    coef1 = coef1[..., None, None]
    coef2 = coef2[..., None, None]
    return E + coef1 * phi_hat + coef2 * phi_hat @ phi_hat

def drct_rotation(drct):
    B,P = drct.shape[0], drct.shape[1]
    drct_n = drct / torch.norm(drct, dim=-1, keepdim=True)
    dummy = torch.tensor([[[0.,0.,1.]]], device=drct.device)#1,1,3
    inner = torch.clamp((drct_n * dummy).sum(-1), min=-1, max=1)
    theta = torch.acos(inner)#B,P
    cross_drct = torch.cross(dummy, drct_n, dim=-1)
    cross_drct /= torch.norm(cross_drct,dim=-1,keepdim=True)#B,P,3
    pos_r = ExpSO3(cross_drct * theta.unsqueeze(-1))
    pos_dummy = torch.einsum('bpij, bpjk -> bpik', pos_r, dummy.unsqueeze(-1))
    pos_distance = torch.norm(pos_dummy.squeeze(-1) - drct_n, dim=-1)
    neg_r = ExpSO3(-cross_drct * theta.unsqueeze(-1))
    neg_dummy = torch.einsum('bpij, bpjk -> bpik', neg_r, dummy.unsqueeze(-1))
    neg_distance = torch.norm(neg_dummy.squeeze(-1) - drct_n, dim=-1)

    ret = []
    for b in range(B):
        for p in range(P):
            if neg_distance[b,p] < pos_distance[b,p]:
                ret.append(neg_r[b,p])
            else:
                ret.append(pos_r[b,p])
    ret = torch.stack(ret, dim=0).reshape(B,P,3,3)
    return ret


def main(path:str):
    instance_name_list = glob(os.path.join('dataset','laptop_output','test', '*', '*', 'input_2', '*.npz'))
    viz = np.load('log/laptop_test/model_20241220_175736/viz/00000.npz', allow_pickle=True)['arr_0'].item()
    #対応するテストデータ
    f = np.load(instance_name_list[viz['idx']], allow_pickle=True)['arr_0'].item()
    #print(f)
    #f = np.load("test.npz", allow_pickle=True)['arr_0'].item()
    
    f = np.load(path, allow_pickle=True)['arr_0'].item()
    seg = f['segmentation']
    det = seg != 0
    #det = f['detection']
    pc = f['pc'][det>0,:]
    print(max(pc[:,0]), min(pc[:,0]))
    print(f['pc'].shape, f['color'].shape, f['segmentation'].shape, f['part'].shape, f['joint'].shape)
    
    color = f['color'][det>0,:]
    seg = seg[det>0]
    directon, pivot = f['joint'][0,:3], f['joint'][0,3:6]
    pose_r, pose_t = f['part'][:,:9].reshape(-1,3,3), f['part'][:,9:12].reshape(-1,3)

    joint_rotation = drct_rotation(torch.tensor(pivot, dtype=torch.float32).reshape(1,1,3)).reshape(3,3)
    print(pivot)
    choise = np.random.choice(pc.shape[0], 1000, replace=True)
    pc = pc[choise]
    color = color[choise]
    seg = seg[choise]
    """
    for i in range(seg.shape[0]):
        if seg[i]==1:
            #color[i,0] = 0.75 + 0.25*color[i,0]
            color[i,0], color[i,1], color[i,2] = 1, 0, 0
        elif seg[i]==2:
            #color[i,1] = 0.75 + 0.25*color[i,1]
            color[i,0], color[i,1], color[i,2] = 0, 1, 0
    """

    j = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.0075, cone_radius=0.015, cylinder_height=0.3, cone_height=0.075)
    j.rotate([[1,0,0],[0,1,0],[0,0,-1]], center=False)
    j.rotate(joint_rotation, center=False)
    j.translate(directon)
    j.paint_uniform_color([0,0,0])

    coors = []
    for i in range(pose_r.shape[0]):
        c = o3d.geometry.TriangleMesh.create_coordinate_frame(2)
        c.rotate(pose_r[i], center=False)
        c.translate(pose_t[i])
        coors.append(c)

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pc)
    o3d_pc.colors = o3d.utility.Vector3dVector(0.85*color)

    #o3d.visualization.draw_geometries([o3d_pc,j,*coors], window_name="viz")
    o3d.visualization.draw_geometries([o3d_pc], window_name="viz")

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    
def create_viz(color_png_path: str, pc_pcd_path: str, seg_npy_path: str):
    color = cv2.imread(color_png_path)
    pc = o3d.io.read_point_cloud(pc_pcd_path)
    segmentation = np.load(seg_npy_path)
    
    pc = np.asarray(pc.points)*10
    color = color.reshape(-1, 3)/255
    #depthが0の点をmaskから除外(realsenseで遠いor近い点のdepthを0にした)
    for i in range(len(pc)):
        if (pc[i,0]==0 and pc[i,1]==0 and pc[i,2]==0):
            segmentation[i] = 0
    valid_indices = segmentation > 0
    pc_seg = pc[valid_indices,:]
    color_seg = color[valid_indices,:]
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pc_seg)
    o3d_pc.colors = o3d.utility.Vector3dVector(color_seg)
    
    print("Radius oulier removal")
    #pc_voxel_down = o3d_pc.voxel_down_sample(voxel_size=0.02)
    #o3d.visualization.draw_geometries([pc_voxel_down])
    cl, ind = o3d_pc.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=1.5)
    display_inlier_outlier(o3d_pc, ind)
    #o3d.visualization.draw_geometries([cl], window_name="viz")
    
    #307200サイズでのsegmentationをつくる
    seg_count = 0
    ind_count = 0
    seg_removed = [0] * segmentation.shape[0]
    for i in range(segmentation.shape[0]):
        if segmentation[i]==1:
            if ind[ind_count] == seg_count:
                seg_removed[i] = 1
                ind_count += 1
                if ind_count >= len(ind):
                    break
            else:
                seg_removed[i] = 0
            seg_count += 1
        else:
            seg_removed[i] = 0
    seg_removed = np.array(seg_removed)
    
    pc_removed = pc[seg_removed>0] 
    color_removed = color[seg_removed>0]
    o3d_pc_removed = o3d.geometry.PointCloud()
    o3d_pc_removed.points = o3d.utility.Vector3dVector(pc_removed)
    o3d_pc_removed.colors = o3d.utility.Vector3dVector(color_removed)
    o3d.visualization.draw_geometries([o3d_pc_removed], window_name="removed")
    
    
    detection = seg_removed != 0
    part = np.random.rand(2, 15)#trainingしないからなんでもいい
    joint = np.random.rand(1, 6)#trainingしないからなんでもいい
    print(pc.shape, color.shape, detection.shape, seg_removed.shape, part.shape, joint.shape) 
    data = {
        'pc': pc,
        'color': color,
        'detection': detection,
        'segmentation': seg_removed,
        'part': part,
        'joint': joint,
    }
    
    # npzファイルとして保存
    output_path = 'dataset/laptop_output/test/gaming/60_desk/input_2/test_open_6.npz'
    np.savez(output_path, arr_0=data)
    

if __name__ == "__main__":
    main('dataset/laptop_output/test/gaming/60_desk/input_2/1702993251.047904961.npz')
    #viz_without_npz()
    #create_viz("../librealsense/wrappers/python/examples/outputs/aligned_rgb.png", "../librealsense/wrappers/python/examples/outputs/from_depth_without_o3d.pcd", "../Grounded-SAM-2/outputs/test_sam2.1/mask.npy")
    