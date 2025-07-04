import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as sciR

import os
from glob import glob

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
    f = np.load(path, allow_pickle=True)['arr_0'].item()   
    #print(f.keys(), f['idx'])

    input = f['input'][0].transpose(1,0)
    inputa = o3d.geometry.PointCloud()
    inputa.points = o3d.utility.Vector3dVector(input)
    
    #print(max(input[:,0]), min(input[:,0]),max(input[:,1]), min(input[:,1]), max(input[:,2]), min(input[:,2]))
    
    
    
    input_r = sciR.random().as_matrix()
    input_align = f['input_align'][0].transpose(1,0)
    print(max(input_align[:,2]) - min(input_align[:,2]))
    input_seg = f['input_seg'][0]
    input_color = torch.zeros_like(input_align) * 0.5
    input_color[:,0], input_color[:,1], input_color[:,2] = input_seg[0,:], input_seg[1,:], 0 if input_seg.shape[0]== 2 else input_seg[2,:]
    #input_color[input_seg.argmax(0)==0,0], input_color[input_seg.argmax(0)==1,1], input_color[input_seg.argmax(0)==2,2] = 1,1,1
    input_joint = f['input_joint'][0]
    input_drct = f['input_drct'][0]
    input_joints, coors = [], []
    for i in range(input_joint.shape[0]):
        joint_rotation = drct_rotation(input_drct.unsqueeze(0)).squeeze(0)
        j = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1)
        j.rotate(joint_rotation[i])
        j.translate(input_joint[i])
        #j.translate([0,-0.1,0])
        #j.rotate(input_r, center=False)
        j.paint_uniform_color([0.25,0.25,0.25])
        input_joints.append(j)
    input_pcl = o3d.geometry.PointCloud()
    input_pcl.points = o3d.utility.Vector3dVector(input_align)
    input_pcl.colors = o3d.utility.Vector3dVector(input_color)
    #input_pcl.rotate(input_r, center=False)

    align2_0 = f['input_align_2'][0,0].transpose(1,0)
    #print(align2_0)
    
    align2_0_color = torch.ones_like(input_align) * 0.75
    align2_0_color[:,0] = torch.maximum(input_seg[0,:], align2_0_color[:,0])
    align2_0_color[align2_0_color[:,0]!=0.75,1], align2_0_color[align2_0_color[:,0]!=0.75,2] = 0, 0
    align2_0_pcl = o3d.geometry.PointCloud()
    align2_0_pcl.points = o3d.utility.Vector3dVector(align2_0)
    align2_0_pcl.colors = o3d.utility.Vector3dVector(align2_0_color)

    align2_1 = f['input_align_2'][0,1].transpose(1,0)
    align2_1_color = torch.ones_like(input_align) * 0.75
    align2_1_color[:,1] = torch.maximum(input_seg[1,:], align2_1_color[:,1])
    align2_1_color[align2_1_color[:,1]!=0.75,0], align2_1_color[align2_1_color[:,1]!=0.75,2] = 0, 0
    align2_1_pcl = o3d.geometry.PointCloud()
    align2_1_pcl.points = o3d.utility.Vector3dVector(align2_1)
    align2_1_pcl.colors = o3d.utility.Vector3dVector(align2_1_color)

    #align2_2 = f['input_align_2'][0,3].transpose(1,0)
    #align2_2_color = torch.ones_like(input_align) * 0.75
    #align2_2_color[:,2] = torch.maximum(input_seg[2,:], align2_2_color[:,2])
    #align2_2_color[align2_2_color[:,2]!=0.75,0], align2_2_color[align2_2_color[:,2]!=0.75,1] = 0, 0
    #align2_2_pcl = o3d.geometry.PointCloud()
    #align2_2_pcl.points = o3d.utility.Vector3dVector(align2_2)
    #align2_2_pcl.colors = o3d.utility.Vector3dVector(align2_2_color)

    recon = f['recon'][0].transpose(1,0)
    recon_seg = f['recon_seg'][0]
    recon_color = torch.zeros_like(input_align) * 0.5
    recon_color[:,0], recon_color[:,1], recon_color[:,2] = recon_seg[0,:], recon_seg[1,:], 0 if recon_seg.shape[0]== 2 else recon_seg[2,:]
    #input_color[input_seg.argmax(0)==0,0], input_color[input_seg.argmax(0)==1,1], input_color[input_seg.argmax(0)==2,2] = 1,1,1
    recon_joint = f['recon_joint'][0]
    recon_drct = f['recon_drct'][0]
    recon_joints, coors = [], []
    for i in range(input_joint.shape[0]):
        joint_rotation = drct_rotation(recon_drct.unsqueeze(0)).squeeze(0)
        j = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1)
        j.rotate(joint_rotation[i])
        j.translate(recon_joint[i])
        #j.translate([0,-0.1,0])
        j.paint_uniform_color([0.25,0.25,0.25])
        recon_joints.append(j)
    recon_pcl = o3d.geometry.PointCloud()
    recon_pcl.points = o3d.utility.Vector3dVector(recon)
    recon_pcl.colors = o3d.utility.Vector3dVector(recon_color)

    o3d.visualization.draw_geometries([inputa], window_name="input")
    o3d.visualization.draw_geometries([input_pcl, *recon_joints],window_name=path+"object level align")
    o3d.visualization.draw_geometries([align2_0_pcl, align2_1_pcl, *recon_joints], window_name="part level align")
    o3d.visualization.draw_geometries([recon_pcl, *recon_joints],window_name="recon")

if __name__ == "__main__":
    dataset = "dataset/pc/partial/safe/test/101363_50p_joint_0.pt"
    instance_name_list = glob(os.path.join('dataset','pc','partial', 'safe', 'test', '*.pt'))  
    index = -1 
    if dataset in instance_name_list:
        index = instance_name_list.index(dataset)
    #print("index:",index)
    file_paths = glob("/home/hasegawa/research/efficient_manip/OP_Align/log/safe_test/model_20250703_185510/viz/102311000.npz")
    for file_path in file_paths:
        #print(file_path)
        main(file_path)