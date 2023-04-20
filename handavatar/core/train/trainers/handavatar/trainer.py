import os

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from handavatar.third_parties.lpips import LPIPS

from handavatar.core.train import create_lr_updater
from handavatar.core.data import create_dataloader
from handavatar.core.utils.network_util import set_requires_grad
from handavatar.core.utils.train_util import cpu_data_to_gpu, Timer
from handavatar.core.utils.image_util import tile_images, to_8b_image
from handavatar.core.utils.metric_util import ssim_metric

from handavatar.configs import cfg
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, BlendParams, MeshRenderer, MeshRasterizer, SoftSilhouetteShader, PointsRasterizationSettings, PointsRasterizer, AlphaCompositor, PointsRenderer
from leap.tools.libmesh import check_mesh_contains
import trimesh

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']


def _unpack_imgs(rgbs, alpha, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    if alpha is not None:
        patch_alpha = torch.zeros_like(patch_imgs[..., 0])
        for i in range(N_patch):
            patch_alpha[i, patch_masks[i]] = alpha[div_indices[i]:div_indices[i+1]]
    else:
        patch_alpha = None

    return patch_imgs, patch_alpha


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


class Trainer(object):
    def __init__(self, network, optimizer, board=None):
        print('\n********** Init Trainer ***********')

        network = network.cuda().deploy_mlps_to_secondary_gpus()
        self.network = network

        self.optimizer = optimizer
        self.update_lr = create_lr_updater()

        if (cfg.resume or cfg.phase=='val') and Trainer.ckpt_exists(cfg.load_net):
            self.load_ckpt(f'{cfg.load_net}')
        elif cfg.phase=='train':
            self.iter = 0
            self.save_ckpt('init')
            self.iter = 1
        else:
            self.iter = 1

        self.timer = Timer()
        self.board = board

        if "lpips" in cfg.train.lossweights.keys():
            self.lpips = LPIPS(net='vgg')
            set_requires_grad(self.lpips, requires_grad=False)
            self.lpips = nn.DataParallel(self.lpips).cuda()
        
        # if "sil" in cfg.train.lossweights.keys():
        self.Rr = torch.tensor([[[-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]]]).float()
        raster_settings_silhouette = PointsRasterizationSettings(
            image_size=(256, 256), 
            radius=0.025,
            bin_size=None,
            points_per_pixel=50,
        )
        self.pcRender = PointsRenderer(
            rasterizer=PointsRasterizer(
            # cameras=cameras, 
            raster_settings=raster_settings_silhouette
            ),
            compositor=AlphaCompositor(background_color=None)
        )

        print("Load Progress Dataset ...")
        self.prog_dataloader = create_dataloader(data_type='progress')

        print('************************************')

    @staticmethod
    def get_ckpt_path(name):
        return os.path.join(cfg.logdir, f'{name}.tar')

    @staticmethod
    def ckpt_exists(name):
        return os.path.exists(Trainer.get_ckpt_path(name))

    ######################################################3
    ## Training 

    def get_img_rebuild_loss(self, loss_names, rgb, target, alpha, target_alpha, **kwargs):
        losses = {}

        if 'mse' in loss_names:
            losses['mse'] = img2mse(rgb, target)

        if 'l1' in loss_names:
            losses['l1'] = img2l1(rgb, target)
        
        if 'iou' in loss_names:
            losses['iou'] = (1. - (alpha.clip(0,1) * target_alpha).view(alpha.shape[0], -1).sum(1) / (1e-5 + alpha.clip(0,1) + target_alpha - alpha.clip(0,1) * target_alpha).abs().view(alpha.shape[0], -1).sum(1)).mean()

        if 'lpips' in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses['lpips'] = torch.mean(lpips_loss)

        return losses
    
    def get_mesh_loss(self, loss_names, target_alpha_img, cam_R, cam_T, cam_K, target_joints, valid_joints, pred_occ, pts, all_out, smpl_output):
        losses = {}
    
        verts = smpl_output['vertices']
        mesh_mano = Meshes(
            verts=verts,
            faces=self.network.smpl_body.faces_tensor[None].repeat(verts[:, :-1].size(0), 1, 1).to(verts.device),
        )
        if 'lap' in loss_names:
            lap = mesh_laplacian_smoothing(mesh_mano)
            losses['lap'] = lap

        if 'nc' in loss_names:
            nc = mesh_normal_consistency(mesh_mano)
            losses['nc'] = nc

        if 'sil' in loss_names:            
            Rs = torch.bmm(self.Rr.to(cam_R.device), cam_R[None].float())
            Ts = torch.bmm(self.Rr.to(cam_R.device), cam_T[None, :, None].float())[..., 0] #torch.tensor([[0, 0, 0],]).float().to(verts.device)
            verts_list = [verts[i] for i in range(verts.shape[0]) ]
            features = [torch.ones(verts.shape[1], 1, device=verts.device) for _ in range(verts.shape[0])]
            cameras = PerspectiveCameras(
                        cam_K[None, [0, 1],[0,1]].float(),
                        cam_K[None, :2, 2].float(),
                        Rs,
                        Ts,
                        in_ndc=False,
                        image_size=[(256, 256)]
                        )
            self.pcRender.rasterizer.cameras = cameras
            self.pcRender.to(verts.device)
            silhouette = self.pcRender(
                Pointclouds(
                points=verts_list, 
                features=features))[..., 0]
            
            radius = self.pcRender.rasterizer.raster_settings.radius
            radius = int(np.round(radius/2. * 256.0 / 1.2))
            target_alpha_img = torch.nn.functional.max_pool2d(target_alpha_img[None], kernel_size=2*radius+1, stride=1, padding=radius)

            losses['sil'] = (1. - (silhouette * target_alpha_img).view(silhouette.shape[0], -1).sum(1) / (1e-5 + silhouette + target_alpha_img - silhouette * target_alpha_img).abs().view(silhouette.shape[0], -1).sum(1)).mean()
        
        if 'lmk' in loss_names:
            joints = smpl_output['joints']
            valid_mask = valid_joints == 1
            losses['lmk'] = img2l1(joints[0, valid_mask], (target_joints-target_joints[4:5])[valid_mask])
            if not valid_mask[4]:
                losses['lmk'] *= 0
        
        if 'iou3d' in loss_names:
            pts = pts.reshape(-1, 3).cpu().numpy()
            all_out = all_out.reshape(-1)
            pred_occ = pred_occ.reshape(-1)
            mesh = trimesh.Trimesh(verts.squeeze().detach().cpu().numpy(), self.network.smpl_body.faces_seal, process=False)
            gt_occ = torch.from_numpy(check_mesh_contains(mesh, pts[~all_out.cpu().numpy()]).astype(np.float32)).to(pred_occ.device)
            losses['iou3d'] = img2mse(pred_occ[~all_out], gt_occ)

        return losses

    def get_loss(self, net_output, patch_masks, bgcolor, targets, 
                    div_indices, target_alpha, target_alpha_img, cam_R, cam_T, cam_K,
                    target_joints, valid_joints):

        lossweights = cfg.train.lossweights
        loss_names = list(lossweights.keys())

        rgb = net_output.pop('rgb')
        if 'alpha' in net_output:
            alpha = net_output.pop('alpha', None)
        else:
            alpha = None
        rgb, alpha = _unpack_imgs(rgb, alpha, patch_masks, bgcolor,
                                     targets, div_indices)
        losses = self.get_img_rebuild_loss(
                        loss_names, 
                        rgb,
                        targets,
                        alpha,
                        target_alpha,
                        **net_output)

        if net_output.get('smpl_output') is not None:
            losses_mesh = self.get_mesh_loss(
                loss_names, 
                target_alpha_img, 
                cam_R, cam_T, cam_K,
                target_joints, valid_joints,
                net_output.get('pred_occ'), net_output.get('pts'), net_output.get('all_out'),
                net_output['smpl_output'])
            losses.update(losses_mesh)

        train_losses = [
            weight * losses[k] for k, weight in lossweights.items()
        ]

        return sum(train_losses), \
               {loss_names[i]: train_losses[i] for i in range(len(loss_names))}

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1

        self.network.train()
        cfg.perturb = cfg.train.perturb

    def train_end(self):
        pass

    def train(self, epoch, train_dataloader):
        self.train_begin(train_dataloader=train_dataloader)

        self.timer.begin()
        for batch_idx, batch in enumerate(train_dataloader):
            if self.iter > cfg.train.maxiter:
                break

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            net_output = self.network(**data)

            train_loss, loss_dict = self.get_loss(
                net_output=net_output,
                patch_masks=data['patch_masks'],
                bgcolor=data['bgcolor'] / 255.,
                targets=data['target_patches'],
                div_indices=data['patch_div_indices'],
                target_alpha=data.get('target_alpha_patches', None),
                target_alpha_img=data.get('target_alpha_img', None),
                target_joints=data.get('dst_cam_joints', None),
                valid_joints=data.get('dst_valid_joints', None),
                cam_R=data['cam_R'],
                cam_T=data['cam_T'],
                cam_K=data['cam_K'],)
            train_loss = train_loss / cfg.train.accum_iter

            train_loss.backward()
            if ((batch_idx + 1) % cfg.train.accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
                self.optimizer.step()
                self.optimizer.zero_grad()
            # self.optimizer.step()

            if self.iter % cfg.train.log_interval == 0:
                loss_str = f"Loss: {cfg.train.accum_iter*train_loss.item():.4f} ["
                for k, v in loss_dict.items():
                    loss_str += f"{k}: {v.item():.4f} "
                loss_str += "]"

                log_str = cfg.experiment + ', Epoch: {} [Iter {}, {}/{} ({:.0f}%), {}] {}'
                log_str = log_str.format(
                    epoch, self.iter,
                    batch_idx * cfg.train.batch_size, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), 
                    self.timer.log(),
                    loss_str)
                print(log_str)
                if self.board is not None:
                    board_dict = loss_dict.copy()
                    if hasattr(self.network, 'density'):
                        board_dict.update({'beta': self.network.density.get_beta().detach()})
                    if 's_val' in net_output:
                        board_dict.update({'s_val': net_output['s_val'].detach().mean().item()})
                    self.board.board_scalar('train', self.iter, self.optimizer.param_groups[0]['lr'], **board_dict)

            is_reload_model = False
            if self.iter % cfg.progress.dump_interval == 0:#or self.iter in [1,300,]:
                is_reload_model = self.progress()

            if not is_reload_model:
                if self.iter % cfg.train.save_checkpt_interval == 0:
                    self.save_ckpt('latest')

                if cfg.save_all:
                    if self.iter % cfg.train.save_model_interval == 0:
                        self.save_ckpt(f'iter_{self.iter}')

                self.update_lr(self.optimizer, self.iter, loader_len=len(train_dataloader))

                self.iter += 1
    
    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        cfg.perturb = 0.
        torch.set_grad_enabled(False)

    def progress_end(self):
        self.network.train()
        cfg.perturb = cfg.train.perturb
        torch.set_grad_enabled(True)

    def progress(self):
        self.progress_begin()

        print('Evaluate Progress Images ...')

        images = []
        psnr = []
        lpips = []
        ssim = []
        is_empty_img = False
        for step, batch in enumerate(self.prog_dataloader):
            if step%100==0:
                print(f'{step} / {len(self.prog_dataloader)}' )
            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.network(**data)

            rendered = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            rgb = net_output['rgb'].data.to("cpu").numpy()
            rendered[ray_mask] = rgb
            rendered = to_8b_image(rendered.reshape((height, width, -1)))

            truth = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            target_rgbs = batch['target_rgbs']
            truth[ray_mask] = target_rgbs
            truth = to_8b_image(truth.reshape((height, width, -1)))
            cat_list = [truth, rendered]

            if 'albedo' in net_output:
                rendered_albedo = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
                albedo = net_output['albedo'].data.to("cpu").numpy()
                rendered_albedo[ray_mask] = albedo
                rendered_albedo = to_8b_image(rendered_albedo.reshape((height, width, -1)))
                cat_list.append(rendered_albedo)

            if 'shadow' in net_output:
                rendered_shadow = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)*0., 
                        dtype='float32')
                shadow = net_output['shadow'].data.to("cpu").numpy()
                rendered_shadow[ray_mask] = shadow[..., None].repeat(3, axis=-1)
                rendered_shadow = to_8b_image(rendered_shadow.reshape((height, width, -1)))
                cat_list.append(rendered_shadow)
            
            if 'specular' in net_output:
                rendered_specular = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)*0., 
                        dtype='float32')
                specular = net_output['specular'].data.to("cpu").numpy()
                rendered_specular[ray_mask] = specular if specular.shape[-1]==3 else specular[..., None].repeat(3, axis=-1)
                rendered_specular = to_8b_image(rendered_specular.reshape((height, width, -1)))
                cat_list.append(rendered_specular)

            truth_alpha = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)*0., 
                        dtype='float32')
            target_alpha = batch['target_alpha']
            truth_alpha[ray_mask] = target_alpha[:, None].tile(1, 3)
            truth_alpha = to_8b_image(truth_alpha.reshape((height, width, -1)))
            cat_list.append(truth_alpha)

            if 'alpha' in net_output:
                rendered_alpha = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)*0., 
                        dtype='float32')
                alpha = net_output['alpha'].data.to("cpu").numpy()
                rendered_alpha[ray_mask] = alpha[..., None].repeat(3, axis=-1)
                rendered_alpha = to_8b_image(rendered_alpha.reshape((height, width, -1)))
                cat_list.append(rendered_alpha)
            
            if 'smpl_output' in net_output:
                verts = net_output['smpl_output']['vertices']
                Rs = torch.bmm(self.Rr.to(data['cam_R'].device), data['cam_R'][None].float())
                Ts = torch.bmm(self.Rr.to(data['cam_T'].device), data['cam_T'][None, :, None].float())[..., 0]
                verts_list = [verts[i] for i in range(verts.shape[0]) ]
                features = [torch.ones(verts.shape[1], 1, device=verts.device) for _ in range(verts.shape[0])]

                cameras = PerspectiveCameras(
                            data['cam_K'][None, [0, 1],[0,1]].float(),
                            data['cam_K'][None, :2, 2].float(),
                            Rs,
                            Ts,
                            in_ndc=False,
                            image_size=[(256, 256)]
                            ).to(verts.device)
                self.pcRender.rasterizer.cameras = cameras
                self.pcRender.to(verts.device)
                silhouette = self.pcRender(
                    Pointclouds(
                    points=verts_list, 
                    features=features))[0, ..., 0]
                silhouette = to_8b_image(silhouette[..., None].tile(1, 1, 3).cpu().numpy())
                
                radius = self.pcRender.rasterizer.raster_settings.radius
                radius = int(np.round(radius/2. * 256.0 / 1.2))
                target_alpha_img = data['target_alpha_img'][None]
                target_alpha_img = torch.nn.functional.max_pool2d(target_alpha_img, kernel_size=2*radius+1, stride=1, padding=radius)[0]
                target_alpha_img = to_8b_image(target_alpha_img[..., None].tile(1, 1, 3).cpu().numpy())
                cat_list += [target_alpha_img, silhouette]

            if 'normal' in net_output:
                rendered_normal = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
                normal = net_output['normal'].data.to("cpu").numpy()
                rendered_normal[ray_mask] = normal
                rendered_normal = to_8b_image(rendered_normal.reshape((height, width, -1)))
                cat_list.append(rendered_normal)
            
            if 'depth' in net_output:
                rendered_depth = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)*0., 
                        dtype='float32')
                depth = net_output['alpha'].data.to("cpu").numpy()
                rendered_depth[ray_mask] = depth[..., None].repeat(3, axis=-1)
                rendered_depth = to_8b_image(rendered_depth.reshape((height, width, -1)))
                cat_list.append(rendered_depth)
            
            images.append(np.concatenate(cat_list, axis=1))

            # check if we create empty images (only at the begining of training)
            if self.iter <= 5000 and np.allclose(rendered, np.array(cfg.bgcolor), atol=5.):
                is_empty_img = True

            rendered = torch.from_numpy(rendered.astype('float32') / 255.0)[None].permute(0, 3, 1, 2)
            truth = torch.from_numpy(truth.astype('float32') / 255.0)[None].permute(0, 3, 1, 2)
            psnr.append(20.0 * torch.log10(1.0 / torch.sqrt(((rendered - truth)**2).mean())))
            lpips.append(self.lpips(
                scale_for_lpips(rendered), 
                scale_for_lpips(truth)))
            ssim.append(ssim_metric(rendered, truth, val_range=1.0))

            if is_empty_img:
                break
            
        bs = np.arange(len(images))
        ch = np.random.choice(bs, min(16, len(images)), replace=False)
        images = [images[i] for i in ch]
        tiled_image = tile_images(images, imgs_per_row=2)
        psnr = sum(psnr)/len(psnr)
        lpips = sum(lpips)/len(lpips)
        ssim = sum(ssim)/len(ssim)

        if self.board is not None:
            self.board.board_img('prog', self.iter, tiled_image)
            self.board.board_scalar('prog', self.iter, **{'psnr': psnr.item(), 'lpips': lpips.item(), 'ssim': ssim.item()})
        else:
            save_dir = os.path.join(cfg.logdir, 'val')
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(tiled_image).save(
            os.path.join(save_dir, "prog_{:06}.jpg".format(self.iter)))
            print('psnr: ', psnr.item(), 'lpips: ', lpips.item(), 'ssim: ', ssim.item())
            
        if is_empty_img:
            print("Produce empty images; reload the init model.")
            self.load_ckpt('init')
            
        self.progress_end()

        return is_empty_img


    ######################################################3
    ## Utils

    def save_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Save checkpoint to {path} ...")

        torch.save({
            'iter': self.iter,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path, map_location='cuda:0')
        self.iter = ckpt['iter'] + 1

        self.network.load_state_dict(ckpt['network'], strict=True)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])
