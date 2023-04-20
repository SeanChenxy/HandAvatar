from configs import cfg, args
import os

import torch
import numpy as np
from tqdm import tqdm

from handavatar.core.data import create_dataloader, create_dataset
from handavatar.core.nets import create_network
from handavatar.core.utils.train_util import cpu_data_to_gpu
from handavatar.core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image
from handavatar.core.utils.rend_util import Renderer
import pyrender
import trimesh

EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height', 'ray_mask']


def load_network(load=True):
    model = create_network()
    if load:
        ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
        ckpt = torch.load(ckpt_path, map_location='cuda:0')
        model.load_state_dict(ckpt['network'], strict=True)
        print('load network from ', ckpt_path)
    return model.cuda().deploy_mlps_to_secondary_gpus()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha=None, shadow=None, specular=None, albedo=None, truth=None, depth=None, a_channel=False):
    
    if alpha is not None:
        alpha_image = np.full((height * width, 3), bgcolor*0, dtype='float32')
        alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
        alpha_image  = to_8b3ch_image(alpha_map)
    else:
        alpha_image = None

    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))
    if a_channel and alpha_image is not None:
        rgb_image = np.concatenate([rgb_image, alpha_image[..., 0:1]], -1)

    if truth is not None:
        truth_image = np.full((height * width, 3), bgcolor, dtype='float32')
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))
    else:
        truth_image = None
    
    if shadow is not None:
        shadow_image = np.full((height * width, 3), bgcolor, dtype='float32')
        # shadow_map = unpack_alpha_map(shadow, ray_mask, width, height)
        # shadow_image  = to_8b3ch_image(shadow_map)
        shadow = np.tile(shadow[:, None], [1,3])
        shadow = shadow + (1.-alpha[:, None]) * bgcolor
        shadow_image[ray_mask] = shadow
        shadow_image = to_8b_image(shadow_image.reshape((height, width, 3)))
        if a_channel and alpha_image is not None:
            shadow_image = np.concatenate([shadow_image, alpha_image[..., 0:1]], -1)
    else:
        shadow_image = None
    
    if specular is not None:
        specular_image = np.full((height * width, 3), bgcolor, dtype='float32')
        specular = np.tile(specular[:, None], [1,3])
        specular = specular + (1.-alpha[:, None]) * bgcolor
        specular_image[ray_mask] = specular
        specular_image = to_8b_image(specular_image.reshape((height, width, 3)))
    else:
        specular_image = None
    
    if albedo is not None:
        albedo_image = np.full((height * width, 3), bgcolor, dtype='float32')
        albedo_image[ray_mask] = albedo
        albedo_image = to_8b_image(albedo_image.reshape((height, width, 3)))
        if a_channel and alpha_image is not None:
            albedo_image = np.concatenate([albedo_image, alpha_image[..., 0:1]], -1)
    else:
        albedo_image = None
    
    if depth is not None:
        depth_image = np.full((height * width, 3), bgcolor*0, dtype='float32')
        depth_map = unpack_alpha_map(depth, ray_mask, width, height)
        depth_image  = to_8b3ch_image(depth_map)
    else:
        depth_image = None

    return rgb_image, alpha_image, shadow_image, specular_image, albedo_image, truth_image, depth_image


def mesh_render(mesh, step=0, z=12.0, a_channel=True):
    VIEWER = pyrender.OffscreenRenderer(256, 256)
    scene = pyrender.Scene(ambient_light=[.1, 0.1, 0.1], bg_color=np.array(cfg.bgcolor) / 255.)
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(12), aspectRatio=1)
    node_cam = pyrender.Node(camera=cam, matrix=np.eye(4))
    scene.add_node(node_cam)
    cam_pose = np.array(
        [[1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, -1, -z],
         [0, 0, 0, 1],], dtype='float32')
    scene.set_pose(node_cam, pose=cam_pose)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1500)
    scene.add(light, pose=cam_pose)

    # update scene
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    color, depth = VIEWER.render(scene)
    if a_channel:
        mask = (depth[..., None] > 0).astype('uint8') * 255
        color = np.concatenate([color, mask], -1)
    return color

@torch.no_grad()
def run():
    cfg.perturb = 0.
    device = 'cuda'
    a_channel = True

    model = load_network(load=(True, False)[cfg.get('ckpts') is not None])
    model = model.eval().to(device)
    subjects = cfg.infer.subject
    if not isinstance(subjects, list):
        subjects = [subjects,]
    if len(cfg.infer.frame)==0:
        frame = [[] for _ in subjects]
    else:
        frame = cfg.infer.frame
    for sub_id, sub in enumerate(subjects):
        test_loader = create_dataloader('infer', subject=sub)
        folder_name = os.path.join(str(cfg.interhand.fps), sub, ('ori', 'sel')[len(frame[sub_id])>0])
        writer = ImageWriter(
                    output_dir=os.path.join(cfg.logdir, cfg.load_net),
                    exp_name=folder_name,
                    clear=True)
        for i, batch in enumerate(test_loader):
            if len(frame[sub_id])>0 and i not in frame[sub_id]:
                writer.skip()
                continue
            if os.path.exists(os.path.join(cfg.logdir, cfg.load_net, folder_name, str(i).zfill(6)+'.png')):
                writer.skip()
                continue
            print(i, len(test_loader))
            for k, v in batch.items():
                batch[k] = v[0]

            data = cpu_data_to_gpu(
                        batch,
                        exclude_keys=EXCLUDE_KEYS_TO_GPU)

            with torch.no_grad():
                net_output = model(**data, 
                                iter_val=cfg.eval_iter)

            rgb = net_output['rgb']
            alpha = net_output['alpha']
            shadow = net_output.get('shadow', None)
            albedo = net_output.get('albedo', None)
            specular = net_output.get('specular', None)
            # depth = net_output['depth']
            shaped_verts = net_output.get('shaped_verts', None)
            smpl_output = net_output.get('smpl_output', None)
            
            imgs = []
            if cfg.infer.save_all==2:
                if shaped_verts is not None:
                    shaped_verts = shaped_verts[0].cpu().numpy() * model.smpl_body.scale
                    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype='float32')
                    shaped_verts = np.dot(rot, shaped_verts.T).T
                    shaped_verts_mesh = trimesh.Trimesh(shaped_verts, model.smpl_body.faces)
                    imgs.append(mesh_render(shaped_verts_mesh, a_channel=True))

                    verts = smpl_output.vertices[0].cpu().numpy()
                    verts_mesh = trimesh.Trimesh(verts, model.smpl_body.faces_seal)
                    imgs.append(mesh_render(verts_mesh, a_channel=True))

                    mesh = model.smpl_body.coap.extract_mesh(
                        smpl_output, 
                        max_queries=cfg.chunk,
                        use_mise=True, 
                        get_vertex_colors=cfg.infer.vert_color,
                        soft=cfg.infer.soft)
                    mesh_img = mesh_render(mesh, a_channel=True)
                    imgs.append(mesh_img)
                else:
                    verts, faces = model.extract_geo(
                        data['dst_bbox_min_xyz'], 
                        data['dst_bbox_max_xyz'], 
                        **data)
                    R = data['cam_R'].cpu().numpy()
                    verts = np.dot(R, verts.T).T
                    mesh = [trimesh.Trimesh(verts, faces, vertex_colors=None),]
                    mesh_img = mesh_render(mesh, z=1.2, a_channel=True)
                    imgs.append(mesh_img)
            
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']
            target_rgbs = batch.get('target_rgbs', None)
            target_img = (batch['target_img'].cpu().numpy() * 255.).astype('uint8')
            target_img = np.concatenate([target_img, np.ones_like(target_img[..., 0:1])*255], -1)

            rgb_img, alpha_img, shadow_img, specular_img, albedo_img, truth_img, depth_img = unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
                rgb.data.cpu().numpy(),
                alpha=alpha.data.cpu().numpy(),
                shadow=shadow.data.cpu().numpy() if shadow is not None else None,
                specular=specular.data.cpu().numpy() if specular is not None else None,
                albedo=albedo.data.cpu().numpy() if albedo is not None else None,
                truth=None,
                a_channel=True)

            if cfg.infer.save_all==2:
                imgs += [albedo_img, shadow_img, specular_img, rgb_img, target_img]
            elif cfg.infer.save_all==1:
                imgs += [albedo_img, shadow_img, rgb_img, target_img]
            else:
                imgs = [rgb_img, target_img]

            imgs = [imgs[i] for i in range(len(imgs)) if imgs[i] is not None]
            img_out = np.concatenate(imgs, axis=1)
            writer.append(img_out)
            if cfg.run_geo:
                writer.append_ply(mesh[0])

        writer.finalize()

if __name__ == '__main__':
    run()
