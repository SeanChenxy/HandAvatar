import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import smplx
from smplx.manohd.subdivide import sub_mano
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ikhand.utils as utils
from ikhand import math_np
import ikhand.skeletons as sk
import torch
import pickle
import numpy as np
import trimesh
from ikhand.math_np import convert
from pytorch3d.loss import mesh_laplacian_smoothing, point_mesh_face_distance, mesh_normal_consistency
from pytorch3d.structures import Meshes, Pointclouds
from tensorboardX import SummaryWriter


def sphere_rand_sample(batch_size):
    def sph2euler(theta, phi):
        z = np.cos(theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        return np.stack([x, y, z], axis=-1)

    theta_x = np.arccos(1 - 2 * np.random.uniform(0, 1, size=[batch_size]))
    phi_x = 2 * np.pi * np.random.uniform(0, 1, size=[batch_size])
    x = sph2euler(theta_x, phi_x)

    angle = np.random.uniform(np.pi, size=[batch_size, 1])
    return x * angle

    # theta_y = np.arccos(1 - 2 * np.random.uniform(0, 1, size=[batch_size]))
    # phi_y = 2 * np.pi * np.random.uniform(0, 1, size=[batch_size])
    # y = sph2euler(theta_y, phi_y)
    # z = np.cross(x, y)
    # z = z / np.sqrt((z ** 2).sum())
    # y = np.cross(z, x)
    # y = y / np.sqrt((y ** 2).sum())
    # rot = np.stack([x, y, z], axis=-1)
    # axis_ang = convert(rot, 'rotmat', 'axangle')
    # return axis_ang

def vis(mano, idx, theta, beta, global_r):
    posed_res = mano(
        torch.from_numpy(beta[None]).float(),
        torch.from_numpy(global_r[None]).float(),
        torch.from_numpy(theta[None]).float(),
        return_verts=True)
    mesh = posed_res.vertices[0].numpy()

    mano_temp = trimesh.Trimesh(mesh, mano.faces)
    _ = mano_temp.export(f'~/Downloads/smplx/{idx}.obj')


class MANODataset(Dataset):
    def __init__(self, scale=1, data_path='template/MANO_RIGHT.pkl'):
        with open(data_path, 'rb') as f:
            mano = pickle.load(f, encoding='latin1')
        self.scale = scale
        self.batch_size = 1024
        theta = mano['hands_mean'] + np.einsum('HW, WD -> HD', mano['hands_coeffs'], mano['hands_components'])
        self.theta_axangle = np.reshape(theta, [-1, 15, 3])
        self.theta = math_np.convert(self.theta_axangle, 'axangle', 'quat')
        self.fingers = 'IMLRT'
        self.finger_poses = []
        for finger in self.fingers:
            finger_poses = []
            for joint in '012':
                finger_poses.append(
                    self.theta[:, sk.MANOHand.labels.index(finger + joint)-1]
                )
            finger_poses = np.stack(finger_poses, 1)
            self.finger_poses.append(finger_poses)

        self.indices = [np.arange(self.finger_poses[i].shape[0]) for i in range(5)]
        self.priority = [None] * 5

    def __sample_pose__(self):
        pose = []
        for i in range(5):
            idx = np.random.choice(
                self.indices[i], size=self.batch_size, p=self.priority[i]
            )
            pose.append(self.finger_poses[i][idx])
        pose = np.stack(pose, 1)
        pose = np.reshape(pose, [self.batch_size, -1, 4])
        return pose

    def __getitem__(self, idx):
        rel_quat_a = self.__sample_pose__()
        rel_quat_b = self.__sample_pose__()
        alpha = np.random.uniform(size=[self.batch_size, 1, 1]).astype(np.float32)
        rel_quat = math_np.slerp_batch(rel_quat_a, rel_quat_b, alpha)
        rel_axangle = math_np.convert(rel_quat, 'quat', 'axangle').reshape(-1, 45).astype('float32')
        
        shape = np.random.normal(
            scale=self.scale, size=[self.batch_size, 10]
        ).astype(np.float32)
        global_r = sphere_rand_sample(self.batch_size).astype('float32')
        # theta = self.theta[idx].reshape(45).astype('float32')

        return{'beta': shape, 'global_r': global_r, 'theta': rel_axangle}

    def __len__(self):
        return 1
    
    def get_ori_data(self):
        n = self.theta_axangle.shape[0]
        rel_axangle = self.theta_axangle[:n].reshape(n, -1)
        rel_axangle = torch.from_numpy(rel_axangle).float()
        shape = torch.zeros(n, 10).float()
        global_r = torch.zeros(n, 3).float()

        return{'beta': shape, 'global_r': global_r, 'theta': rel_axangle}

class Model(torch.nn.Module):
    def __init__(self, level, pretrain):
        super(Model, self).__init__()
        sub_times = level
        mano = smplx.create(**smpl_cfg)
        if not smpl_cfg['is_rhand']:
            mano.shapedirs[:,0,:] *= -1
        self.mano, edge, spirals = sub_mano(mano, sub_times, pretrain=pretrain)
        self.edge = torch.from_numpy(edge).long()
        self.spirals = torch.tensor(spirals).long()
        self.mano_ori = smplx.create(**smpl_cfg)
        if not smpl_cfg['is_rhand']:
            self.mano_ori.shapedirs[:,0,:] *= -1
        self.mano.requires_grad_(False)
        self.mano_ori.requires_grad_(False)
        mano_tmp = smplx.create(**smpl_cfg)
        if not smpl_cfg['is_rhand']:
            mano_tmp.shapedirs[:,0,:] *= -1
        mano_tmp, edge, spirals = sub_mano(mano_tmp, sub_times, pretrain=None)
        self.ori_lbs_weights = mano_tmp.lbs_weights.clone()
        self.pre_lbs_weights = mano.lbs_weights.clone()
        weights = mano.lbs_weights.clone()
        self.weights = torch.nn.Parameter(weights)
    
    def forward(self, beta, global_r, theta):
        # lbs_weights = torch.nn.functional.hardtanh(self.weights, min_val=0.0, max_val=1.0)
        lbs_weights = self.weights.clamp(min=0.0, max=1.0)
        lbs_weights = lbs_weights / lbs_weights.sum(-1, keepdim=True)
        # lbs_weights = torch.cat([self.ori_lbs_weights.to(lbs_weights.device), lbs_weights], dim=0)

        mano_res = self.mano(beta, global_r, theta, custom_lbs_weights=lbs_weights, return_verts=True)
        verts = mano_res.vertices

        mano_res = self.mano(beta, global_r, theta, custom_lbs_weights=self.ori_lbs_weights.to(lbs_weights.device), return_verts=True)
        verts_ori = mano_res.vertices

        mano_res = self.mano_ori(beta, global_r, theta, return_verts=True)
        verts_mano = mano_res.vertices

        return verts, verts_ori, verts_mano, lbs_weights


def board_scalar(board, phase, n_iter, lr=None, **kwargs):
    split = '/'
    for key, val in kwargs.items():
        if isinstance(val, torch.Tensor):
            val = val.item()
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, torch.Tensor):
                    val = val.item()
                board.add_scalar(phase + split + key + split + sub_key, sub_val, n_iter)
        elif isinstance(val, tuple):
            for sub_key, sub_val in enumerate(val):
                board.add_scalar(phase + split + key + split + str(sub_key), sub_val, n_iter)
        else:
            board.add_scalar(phase + split + key, val, n_iter)
    if lr:
        board.add_scalar(phase + split + 'lr', lr, n_iter)

def train(model, dataloader, optimizer):
    model.train()
    for step, data in enumerate(dataloader):
        beta = data['beta'][0].to(device)
        global_r = data['global_r'][0].to(device)
        theta = data['theta'][0].to(device)
        # theta[:, 5] = 1.5
        verts, verts_ori, verts_mano, lbs_weights = model(beta, global_r, theta)
        
        loss_w = (model.weights[:model.edge.max()+1] - model.pre_lbs_weights[:model.edge.max()+1].to(verts.device)).abs().mean()

        mesh = Meshes(
            verts=verts,
            faces=torch.from_numpy(model.mano.faces.astype('int64'))[None].repeat(verts.size(0), 1, 1).to(verts.device),
        )

        mesh_mano = Meshes(
            verts=verts_mano,
            faces=torch.from_numpy(model.mano_ori.faces.astype('int64'))[None].repeat(verts_mano.size(0), 1, 1).to(verts_mano.device),
        )

        pc = Pointclouds(verts)
        loss_dist = point_mesh_face_distance(mesh_mano, pc) 

        loss_lap = mesh_laplacian_smoothing(mesh)

        loss_nc = mesh_normal_consistency(mesh)

        loss_l0 = (1 - torch.exp(-100*lbs_weights)).mean()

        loss =  loss_lap + loss_dist * surf_weight + loss_w * preski_weight + loss_l0 * l0_weight + loss_nc * nc_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss_lap, loss_dist, loss_w, loss_l0, loss_nc, verts, verts_ori, verts_mano, lbs_weights.data

def eval(model, dataset):
    model.eval()
    data = dataset.get_ori_data()
    beta = data['beta'].to(device)
    global_r = data['global_r'].to(device)
    theta = data['theta'].to(device)

    chunk = 500
    beta_list = torch.split(beta, chunk, dim=0)
    global_r_list = torch.split(global_r, chunk, dim=0)
    theta_list = torch.split(theta, chunk, dim=0)

    dist = []
    dist_sub = []
    lap = []
    lap_sub = []
    lap_mano = []
    nc = []
    nc_sub = []
    nc_mano = []

    for beta, global_r, theta in zip(beta_list, global_r_list, theta_list):
        verts, verts_ori, verts_mano, lbs_weights = model(beta, global_r, theta)

        mesh = Meshes(
            verts=verts,
            faces=torch.from_numpy(model.mano.faces.astype('int64'))[None].repeat(verts.size(0), 1, 1).to(verts.device),
        )

        mesh_sub = Meshes(
            verts=verts_ori,
            faces=torch.from_numpy(model.mano.faces.astype('int64'))[None].repeat(verts.size(0), 1, 1).to(verts.device),
        )

        mesh_mano = Meshes(
            verts=verts_mano,
            faces=torch.from_numpy(model.mano_ori.faces.astype('int64'))[None].repeat(verts_mano.size(0), 1, 1).to(verts_mano.device),
        )

        pc = Pointclouds(verts)
        pc_sub = Pointclouds(verts_ori)
        dist.append(point_mesh_face_distance(mesh_mano, pc))
        dist_sub.append(point_mesh_face_distance(mesh_mano, pc_sub))

        lap.append(mesh_laplacian_smoothing(mesh))
        lap_sub.append(mesh_laplacian_smoothing(mesh_sub))
        lap_mano.append(mesh_laplacian_smoothing(mesh_mano))

        nc.append(mesh_normal_consistency(mesh))
        nc_sub.append(mesh_normal_consistency(mesh_sub))
        nc_mano.append(mesh_normal_consistency(mesh_mano))

    dist = sum(dist) / len(dist)
    dist_sub = sum(dist_sub) / len(dist_sub)
    lap = sum(lap) / len(lap)
    lap_sub = sum(lap_sub) / len(lap_sub)
    lap_mano = sum(lap_mano) / len(lap_mano)
    nc = sum(nc) / len(nc)
    nc_sub = sum(nc_sub) / len(nc_sub)
    nc_mano = sum(nc_mano) / len(nc_mano)
    L0 = (lbs_weights.abs() > 0).sum()
    L0_sub = (model.ori_lbs_weights.abs() > 0).sum()

    return dist, lap, nc, L0, dist_sub, lap_sub, nc_sub, L0_sub, lap_mano, nc_mano


if __name__ == '__main__':
    smpl_cfg = {}
    smpl_cfg['model_type'] = 'mano'
    smpl_cfg['flat_hand_mean'] = True
    smpl_cfg['use_pca'] = False
    smpl_cfg['center_id'] = 4
    smpl_cfg['scale'] = 10.
    smpl_cfg['seal'] = False
    smpl_cfg['model_path'] = '../../Libs/models'
    scale = 1

    # key param
    smpl_cfg['is_rhand'] = True
    levels = [2,]
    surf_weight = 5
    preski_weight = 0
    l0_weight = 0.01
    nc_weight = 0.01
    max_epoch = 3000
    pretrain = None
    resume = None
    post=''

    dataset = MANODataset(scale=scale, data_path=('template/MANO_LEFT.pkl', 'template/MANO_RIGHT.pkl')[smpl_cfg['is_rhand']])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    for level in levels:
        if len(levels)==2 and level==1 and max_epoch>0:
            pretrain = exp
        model = Model(level, pretrain)
        if max_epoch==-1:
            assert resume is not None
            path = f'smplx/out/{resume}/ckpts/lbs_weights.pth'
            lbs_weights = torch.load(path, map_location='cpu')
            model.weights.data = lbs_weights
            print('Resume from ', path)
        else:
            exp = f'sub{level}_surf{surf_weight}_nc{nc_weight}_reg{l0_weight}_{post}'
            print(exp)
            os.makedirs(f'smplx/out/{exp}', exist_ok=True)
            os.makedirs(f'smplx/out/{exp}/vis', exist_ok=True)
            os.makedirs(f'smplx/out/{exp}/board', exist_ok=True)
            os.makedirs(f'smplx/out/{exp}/ckpts', exist_ok=True)
            board = SummaryWriter(f'smplx/out/{exp}/board')
            
        for key, value in model.named_parameters():
            if value.requires_grad:
                print(key, value.shape)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
        
        device = 'cuda'
        model = model.to(device)
        total_step = 0
        for epoch in range(max_epoch+1):
            loss_lap, loss_dist, loss_w, loss_l0, loss_nc, verts, verts_ori, verts_mano, lbs_weights = train(model, dataloader, optimizer)
            # with torch.no_grad():
            #     dist, lap, L0, dist_sub, lap_sub, L0_sub = eval(model, dataset)

            if total_step % 100 == 0:
                print(epoch, total_step, loss_lap.item(), loss_dist.item(), loss_w.item(), loss_l0.item(), loss_nc.item(), optimizer.param_groups[0]['lr'])
            board_scalar(board, 'train_{}'.format(exp[:4]), total_step, optimizer.param_groups[0]['lr'], 
                **{'lap': loss_lap.item(), 'dist': loss_dist.item(), 'w': loss_w.item(), 'nc': loss_nc.item(), 'L0ap': loss_l0.item(), 'L0': (lbs_weights.abs() > 0).sum().item()})
            # board_scalar(board, 'val', total_step, None, **{
            #     'dist': dist.item(), 'dist_sub': dist_sub.item(), 
            #     'lap': lap.item(), 'lap_sub': lap_sub.item(),
            #     'L0': L0.item(), 'L0_sub': L0_sub.item(),})

            if total_step % 500 == 0:
                torch.save(lbs_weights, f'smplx/out/{exp}/ckpts/lbs_weights_{total_step}.pth')
                torch.save(lbs_weights, f'smplx/out/{exp}/ckpts/lbs_weights.pth')
            
                verts = verts[0].detach().cpu().numpy()
                mesh = trimesh.Trimesh(verts, model.mano.faces)
                _ = mesh.export(f'smplx/out/{exp}/vis/sub_{total_step}.obj')

                verts = verts_mano[0].detach().cpu().numpy()
                mesh = trimesh.Trimesh(verts, model.mano_ori.faces)
                _ = mesh.export(f'smplx/out/{exp}/vis/mano_{total_step}.obj')

                verts = verts_ori[0].detach().cpu().numpy()
                mesh = trimesh.Trimesh(verts, model.mano.faces)
                _ = mesh.export(f'smplx/out/{exp}/vis/ori_{total_step}.obj')
            
            total_step += 1
            scheduler.step()

        with torch.no_grad():
            dist, lap, nc, L0, dist_sub, lap_sub, nc_sub, L0_sub, lap_mano, nc_mano = eval(model, dataset)
            print(dist.item(), dist_sub.item(), lap.item(), lap_sub.item(), nc.item(), nc_sub.item(), L0.item(), L0_sub.item(), lap_mano.item(), nc_mano.item())
