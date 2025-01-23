import torch
import numpy as np
import open3d as o3d

from glob import glob
import os 
from pytorch3d.ops import knn_points

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


def main(f_list, test_data):
    f_list = glob(f_list)
    instance_name_list = glob(os.path.join('dataset','laptop_output','test', '*', '*', 'input_2', '*.npz'))

    for f_i in range(f_list.__len__()):
        
        viz = np.load(f_list[f_i], allow_pickle=True)['arr_0'].item()
        ori = np.load(instance_name_list[viz['idx']], allow_pickle=True)['arr_0'].item()
        
        if instance_name_list[viz['idx']] != test_data:
            continue
        print(viz['idx'])
        
        viz_pc = viz['input'][0].permute(1,0)
        print("max: ", max(viz_pc[:, 0]))
        viz_pc /= viz['gt_expand']
        viz_pc += viz['gt_center']
        print("expand:", viz['gt_expand'], "center:", viz['gt_center'], "max_after:",max(viz_pc[:, 0]), "min_after:",min(viz_pc[:,0]))
        viz_joint = viz['input_joint'][0]
        viz_joint /= viz['gt_expand']
        viz_joint += viz['gt_center'].reshape(1,3)
        print("viz_joint:", viz_joint)
        viz_drct = viz['input_drct'][0]
        viz_seg = viz['input_seg'][0]

        ori_pc = ori['pc']
        ori_color = ori['color']
        print(ori_color.shape)
        #ori_color[ori['detection'],:] *= 2
        ori_det_pc = ori_pc[ori['detection']]
        
        ori_joint = ori['joint'][:,:3]
        ori_drct = ori['joint'][:,3:]

        ori_part_d = []
        for i in range(viz_seg.shape[0]):
            part_pc = viz_pc[torch.max(viz_seg, dim=0).indices == i]
            cur_d = knn_points(torch.tensor(ori_det_pc, dtype=torch.float32).reshape(1,-1,3), part_pc.reshape(1,-1,3)).dists
            ori_part_d.append(cur_d[0,:,0])
        ori_part_d = torch.stack(ori_part_d, dim=0)
        ori_seg = torch.min(ori_part_d, dim=0).indices

        
        ori_color[ori['detection'],0] = np.where(ori_seg==0, 0.5 + 0.5*ori_color[ori['detection'],0], ori_color[ori['detection'],0])
        ori_color[ori['detection'],1] = np.where(ori_seg==0, 0 + 0.5*ori_color[ori['detection'],1], ori_color[ori['detection'],1])
        ori_color[ori['detection'],2] = np.where(ori_seg==0, 0 + 0.5*ori_color[ori['detection'],2], ori_color[ori['detection'],2])

        ori_color[ori['detection'],0] = np.where(ori_seg==1, 0.5*ori_color[ori['detection'],0], ori_color[ori['detection'],0])
        ori_color[ori['detection'],1] = np.where(ori_seg==1, 0.5 + 0.5*ori_color[ori['detection'],1], ori_color[ori['detection'],1])
        ori_color[ori['detection'],2] = np.where(ori_seg==1, 0.5*ori_color[ori['detection'],2], ori_color[ori['detection'],2])

        ori_color[ori['detection'],0] = np.where(ori_seg==2, 0.5*ori_color[ori['detection'],0], ori_color[ori['detection'],0])
        ori_color[ori['detection'],1] = np.where(ori_seg==2, 0.5*ori_color[ori['detection'],1], ori_color[ori['detection'],1])
        ori_color[ori['detection'],2] = np.where(ori_seg==2, 0.5 + 0.5*ori_color[ori['detection'],2], ori_color[ori['detection'],2])

        viz_joints = []
        for i in range(viz_joint.shape[0]):
            joint_rotation = drct_rotation(viz_drct.unsqueeze(0)).squeeze(0)
            #joint_rotation = drct_rotation(torch.tensor(ori_drct, dtype=torch.float).unsqueeze(0)).squeeze(0)
            j = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.05, cone_radius=0.1, cylinder_height=2, cone_height=0.5)
            j.rotate(np.array([[1,0,0],[0,1,0],[0,0,-1]]), center=False)
            j.rotate(joint_rotation[i], center=False)
            j.translate(viz_joint[i])
            #j.translate(ori_joint[i])
            #j.translate(0.5*ori_joint[i] + 0.5*viz_joint[i].numpy())
            j.paint_uniform_color([0,0,0])
            viz_joints.append(j)
            print(j)
            

        non_zero_idx = ori_det_pc.sum(-1)!=0
        ori_det_pc = ori_det_pc[non_zero_idx]
        ori_det_color = ori_color[ori['detection']][non_zero_idx]
        #ori_det_color = ori_color[non_zero_idx]
        o3d_ori_pc = o3d.geometry.PointCloud()
        o3d_ori_pc.points = o3d.utility.Vector3dVector(ori_det_pc)
        o3d_ori_pc.colors = o3d.utility.Vector3dVector(ori_det_color)
        o3d.visualization.draw_geometries([o3d_ori_pc, *viz_joints])
        o3d_viz_pc = o3d.geometry.PointCloud()
        o3d_viz_pc.points = o3d.utility.Vector3dVector(viz_pc)
        #o3d_viz_pc.colors = o3d.utility.Vector3dVector(ori_det_color)
        o3d.visualization.draw_geometries([o3d_viz_pc, *viz_joints])
        
        break
    
def viz_comparison_syn():
    viz = np.load("log/safe-hsaur-101603-30_test/model_20250114_145806/viz/101603003.npz", allow_pickle=True)['arr_0'].item()
    ori = torch.load("dataset/pc/partial/safe-hsaur/101603/test-30/3.pt")
    
    viz_pc = viz['input'][0].permute(1,0)
    print("max: ", max(viz_pc[:,0]), max(viz_pc[:,1]), max(viz_pc[:,2]))
    viz_pc *= (max(ori['pc'].numpy()[0]) / max(viz_pc[0]))
    #viz_pc += viz['gt_center']
    print("expand:", viz['gt_expand'], "center:", viz['gt_center'], "max_after_expand:",max(viz_pc[:,0]),max(viz_pc[:,1]),max(viz_pc[:,2]), "min_after_expand:",min(viz_pc[:,0]),min(viz_pc[:,1]),min(viz_pc[:,2]))
    viz_joint = viz['input_joint'][0]
    viz_joint /= viz['gt_expand']
    #viz_joint += viz['gt_center'].reshape(1,3)
    print("viz_joint:", viz_joint)
    viz_drct = viz['input_drct'][0]
    viz_seg = viz['input_seg'][0]

    ori_pc = ori['pc'].numpy()
    print("ori_max:", max(ori_pc[:,0]), max(ori_pc[:,1]), max(ori_pc[:,2]), "ori_min:", min(ori_pc[:,0]), min(ori_pc[:,1]), min(ori_pc[:,2]))
    ori_color = np.zeros((ori_pc.shape[0], 3))
    #ori_color = ori['color']
    #ori_color[ori['detection'],:] *= 2
    
    #ori_joint = ori['joint'][:,:3]
    #ori_drct = ori['joint'][:,3:]

    ori_part_d = []
    for i in range(viz_seg.shape[0]):
        part_pc = viz_pc[torch.max(viz_seg, dim=0).indices == i]
        cur_d = knn_points(torch.tensor(ori_pc, dtype=torch.float32).reshape(1,-1,3), part_pc.reshape(1,-1,3)).dists
        ori_part_d.append(cur_d[0,:,0])
    ori_part_d = torch.stack(ori_part_d, dim=0)
    ori_seg = torch.min(ori_part_d, dim=0).indices

    ori_color[:,0] = np.where(ori_seg==0, 1 + 0.5*ori_color[:,0], ori_color[:,0])
    ori_color[:,1] = np.where(ori_seg==0, 0 + 0.5*ori_color[:,1], ori_color[:,1])
    ori_color[:,2] = np.where(ori_seg==0, 0 + 0.5*ori_color[:,2], ori_color[:,2])

    ori_color[:,0] = np.where(ori_seg==1, 0.5*ori_color[:,0], ori_color[:,0])
    ori_color[:,1] = np.where(ori_seg==1, 1 + 0.5*ori_color[:,1], ori_color[:,1])
    ori_color[:,2] = np.where(ori_seg==1, 0.5*ori_color[:,2], ori_color[:,2])

    ori_color[:,0] = np.where(ori_seg==2, 0.5*ori_color[:,0], ori_color[:,0])
    ori_color[:,1] = np.where(ori_seg==2, 0.5*ori_color[:,1], ori_color[:,1])
    ori_color[:,2] = np.where(ori_seg==2, 1 + 0.5*ori_color[:,2], ori_color[:,2])

    viz_joints = []
    for i in range(viz_joint.shape[0]):
        joint_rotation = drct_rotation(viz_drct.unsqueeze(0)).squeeze(0)
        #joint_rotation = drct_rotation(torch.tensor(ori_drct, dtype=torch.float).unsqueeze(0)).squeeze(0)
        j = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.05, cone_radius=0.1, cylinder_height=2, cone_height=0.5)
        j.rotate(np.array([[1,0,0],[0,1,0],[0,0,-1]]), center=False)
        j.rotate(joint_rotation[i], center=False)
        j.translate(viz_joint[i])
        #j.translate(ori_joint[i])
        #j.translate(0.5*ori_joint[i] + 0.5*viz_joint[i].numpy())
        j.paint_uniform_color([0,0,0])
        viz_joints.append(j)
        print(j)
            
    o3d_ori_pc = o3d.geometry.PointCloud()
    o3d_ori_pc.points = o3d.utility.Vector3dVector(ori_pc)
    o3d_ori_pc.colors = o3d.utility.Vector3dVector(ori_color)
    o3d.visualization.draw_geometries([o3d_ori_pc, *viz_joints], window_name="ori_comparison")
    o3d_viz_pc = o3d.geometry.PointCloud()
    o3d_viz_pc.points = o3d.utility.Vector3dVector(viz_pc)
    #o3d_viz_pc.colors = o3d.utility.Vector3dVector(ori_det_color)
    o3d.visualization.draw_geometries([o3d_viz_pc, *viz_joints], window_name="viz_comparison")
    
    
    
def check_expand(path):
    viz = np.load(path, allow_pickle=True)['arr_0'].item()
    viz_pc = viz['input'][0].permute(1,0)
    print("max: ", max(viz_pc[:, 0]), viz_pc.shape, viz['gt_center'].shape)
    viz_pc /= viz['gt_expand']
    #viz_pc += viz['gt_center']
    print("expand:", viz['gt_expand'], "center:", viz['gt_center'], "max_after:",max(viz_pc[:, 0]), "min_after:",min(viz_pc[:,0]))
    viz_joint = viz['input_joint'][0]
    viz_joint /= viz['gt_expand']
    viz_joint += viz['gt_center'].reshape(1,3)
    print("viz_joint:", viz_joint)
    viz_drct = viz['input_drct'][0]
    viz_seg = viz['input_seg'][0]


if __name__ == "__main__":
    #main('log/laptop_test/model_20241220_175736/viz/*.npz', 'dataset/laptop_output/test/gaming/60_desk/input_2/test_open_4.npz')
    viz_comparison_syn()
    #check_expand("log/safe-50_test/model_20241231_123155/viz/101363050.npz")