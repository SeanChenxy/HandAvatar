import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import smplx
import trimesh
import torch
import numpy as np
from utils.generate_spiral_seq import _next_ring
from smplx.manohd.subdivide import sub_mano, sphere_rand_sample
from manotorch.axislayer import AxisLayer
from manotorch.manolayer import ManoLayer
from htmlhand.utils.tools import get_theta
from manotorch.utils.quatutils import quaternion_to_angle_axis
import pyrender
import cv2
from coaphand.training_code.renderer import Renderer



def visualize(mesh):
    scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(60), aspectRatio=1)
    node_cam = pyrender.Node(camera=cam, matrix=np.eye(4))
    scene.add_node(node_cam)
    cam_pose = np.array(
        [[1, 0, 0, 0.2],
         [0, 1, 0, 0],
         [0, 0, 1, 2],
         [0, 0, 0, 1],], dtype='float32')
    scene.set_pose(node_cam, pose=cam_pose)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1500)
    scene.add(light, pose=np.eye(4))

    # update scene
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    color, depth = VIEWER.render(scene)
    return color[..., ::-1]


def main(smpl_cfg, freepose_type, key_step, joint_rot, exp_name, device='cuda'):
    # create a SMPL body and attach COAP
    smpl_body = smplx.create(**smpl_cfg)
    mano = smplx.create(**smpl_cfg)
    smpl_body_ori = smplx.create(**smpl_cfg)
    if not smpl_cfg['is_rhand']:
        smpl_body.shapedirs[:,0,:] *= -1
        mano.shapedirs[:,0,:] *= -1
        smpl_body_ori.shapedirs[:,0,:] *= -1

    smpl_body_ori ,_ ,_ = sub_mano(smpl_body_ori, 2)
    smpl_body ,_ ,_ = sub_mano(smpl_body, 2)
    lbs_weights = torch.load(f'smplx/out/{exp_name}/ckpts/lbs_weights.pth', map_location='cpu')
    smpl_body.lbs_weights = lbs_weights

    smpl_body = smpl_body.to(device)
    smpl_body_ori = smpl_body_ori.to(device)
    mano = mano.to(device)

    mano_layer = ManoLayer(
        rot_mode='quat',
        use_pca=False,
        side=('left', 'right')[smpl_cfg['is_rhand']],
        center_idx=0,
        mano_assets_root='template',
        flat_hand_mean=True,)
    axis_layer = AxisLayer()
    rest_pose = torch.zeros(1, 64)
    rest_pose[:, [i for i in range(64) if i%4==0]] = 1
    rest_shape = torch.zeros(1, 10)
    rest_results = mano_layer(rest_pose, rest_shape)
    bul_axes = axis_layer(rest_results.joints, rest_results.transforms_abs)

    data = {}
    data['betas'] = torch.zeros(1, 10).float().to(device)
    data['hand_pose'] = torch.zeros(1, 45).float().to(device)
    angle = np.pi / 2
    global_orient_rmtx = cv2.Rodrigues(np.array([-angle, 0, 0], dtype='float32'))[0]
    global_orient = cv2.Rodrigues(global_orient_rmtx)[0][:, 0]
    data['global_orient'] = torch.from_numpy(global_orient)[None].float().to(device)

    total_step = 100 if freepose_type<3 else freepose_type
    for step in range(total_step):
        print(step, total_step)
        if isinstance(key_step, list) and step not in key_step:
            continue
        # load rand pose
        if freepose_type:
            dst_poses = get_theta(joint_rot, bul_axes, rest_pose, -freepose_type, step).view(16, 4)
            dst_poses = quaternion_to_angle_axis(dst_poses).cpu().numpy().reshape(48).astype('float32')
            data['hand_pose'] = torch.from_numpy(dst_poses[3:])[None].float().to(device)

        # rotate body
        if freepose_type < 3:
            angle = 2 * np.pi / 100 * step
            global_orient_rmtx = cv2.Rodrigues(np.array([-angle, 0, 0], dtype='float32'))[0]
            global_orient = cv2.Rodrigues(global_orient_rmtx)[0][:, 0]
            data['global_orient'] = torch.from_numpy(global_orient)[None].float().to(device)

        print(data['global_orient'])
        # smpl forward pass
        with torch.no_grad():
            smpl_output = smpl_body(**data, return_verts=True, return_full_pose=True)
            smpl_ori_output = smpl_body_ori(**data, return_verts=True, return_full_pose=True)
            mano_output = mano(**data, return_verts=True, return_full_pose=True, distal=True)

        if True:
            verts = smpl_output.vertices[0].cpu().numpy()
            mesh = trimesh.Trimesh(verts, smpl_body.faces_seal)
            verts = smpl_ori_output.vertices[0].cpu().numpy()
            mesh_ori = trimesh.Trimesh(verts, smpl_body.faces_seal)
            verts = mano_output.vertices[0].cpu().numpy()
            # print(mano_output.joints[0].cpu().numpy().tolist())
            import vctoolkit as vc
            from ikhand.skeletons import MANOHand
            joints = mano_output.joints[0].cpu().numpy()
            vc.joints_to_mesh(joints, MANOHand, save_path='test.obj')
            mesh_mano = trimesh.Trimesh(verts, mano.faces_seal)

            mesh.export(os.path.join(SAVE_DIR, str(step).zfill(3) +'.obj'))
            mesh_ori.export(os.path.join(SAVE_DIR, str(step).zfill(3) +'_ori.obj'))
            mesh_mano.export(os.path.join(SAVE_DIR, str(step).zfill(3) +'_mano.obj'))

        rend_mesh = (MESH_RENDER.render_mesh(
                smpl_output.vertices.float().to(device),
                smpl_body.faces_tensor.unsqueeze(0).to(device),
                torch.ones_like(smpl_output.vertices.float()),
                mode='n'
            )[0].squeeze(0) * 255.).cpu().numpy().astype(np.uint8)
        
        rend_mesh_ori = (MESH_RENDER.render_mesh(
                smpl_ori_output.vertices.float().to(device),
                smpl_body_ori.faces_tensor.unsqueeze(0).to(device),
                torch.ones_like(smpl_ori_output.vertices.float()),
                mode='n'
            )[0].squeeze(0) * 255.).cpu().numpy().astype(np.uint8)

        rend_mesh_mano = (MESH_RENDER.render_mesh(
                mano_output.vertices.float().to(device),
                mano.faces_tensor.unsqueeze(0).to(device),
                torch.ones_like(mano_output.vertices.float()),
                mode='n'
            )[0].squeeze(0) * 255.).cpu().numpy().astype(np.uint8)
        
        # rend_mesh = visualize(mesh)
        # rend_mesh_mano = visualize(mesh_mano)
        img = np.concatenate([rend_mesh_mano, rend_mesh_ori, rend_mesh], 1)

        cv2.imwrite(os.path.join(SAVE_DIR, str(step).zfill(3) +'.png'), img)


if __name__ == '__main__':
    smpl_cfg = {}
    smpl_cfg['model_type'] = 'mano'
    smpl_cfg['flat_hand_mean'] = True
    smpl_cfg['use_pca'] = False
    smpl_cfg['is_rhand'] = True
    smpl_cfg['center_id'] = 4
    smpl_cfg['scale'] = 10.
    smpl_cfg['seal'] = True
    smpl_cfg['model_path'] = '../../Libs/models'

    joint_rot = [[0, 0, 0],
              [[0, 90], 0, 0],
              [[0, 80], 0, 0],
              [[0, 80], 0, 0],
              [[0, 100], 0, 0],
              [[0, 80], 0, 0],
              [[0, 80], 0, 0],
              [[0, 90], 0, 0],
              [[0, 80], 0, 0],
              [[0, 80], 0, 0],
              [[0, 90], 0, 0],
              [[0, 80], [0, 10], 0],
              [[0, 80], 0, 0],
              [0, 0, [0, 50]],
              [0, [0, 20], 0],
              [[0, 90], 0, 0]]
    exp_name = 'sub2_surf10_nc0.01_reg0.01_'
    # if not smpl_cfg['is_rhand']:
    #     exp_name += 'left'
    # VIEWER = pyrender.OffscreenRenderer(512, 512)
    freepose_type = [20, [0, 15, 19]]
    key_step = -1
    if isinstance(freepose_type, list):
        freepose_type, key_step = freepose_type

    SAVE_DIR = os.path.join(f'smplx/out/{exp_name}', ('tpose', 'free', 'fist', 'trans')[min(freepose_type, 3)])
    MESH_RENDER = Renderer()

    if os.path.exists(SAVE_DIR):
        os.system('rm -r %s/*' % SAVE_DIR)
    else:
        os.makedirs(SAVE_DIR)

    main(smpl_cfg, freepose_type, key_step, joint_rot, exp_name)
