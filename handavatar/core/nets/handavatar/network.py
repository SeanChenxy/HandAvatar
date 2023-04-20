import torch
import torch.nn as nn
import torch.nn.functional as F

from handavatar.core.nets.handavatar.component_factory import \
    load_positional_embedder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp, \
    load_render_mlp, \
    load_deform_mlp, \
    load_shadow_network

from configs import cfg
import mcubes
import numpy as np
from pairof.pairof_model import attach_pairof
import smplx
from handavatar.core.utils.network_util import set_requires_grad
from smplx.manohd.subdivide import sub_mano
from smplx.utils import vertex_normals
from smplx.lbs import get_normal_coord_system
from leap.tools.libmesh import check_mesh_contains
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from collections import OrderedDict


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # manohd
        smpl_body = smplx.create(**cfg.smpl_cfg).requires_grad_(False)
        if not cfg.smpl_cfg.is_rhand:
            smpl_body.shapedirs[:,0,:] *= -1
        if cfg.smpl_cfg['manohd']>0:
            print('MANO-HD in Model')
            smpl_body ,_ ,_ = sub_mano(smpl_body, cfg.smpl_cfg['manohd'])
            lbs_weights = torch.load(cfg.smpl_cfg['lbs_weights'], map_location='cpu')
            smpl_body.lbs_weights = lbs_weights
        
        # pairof
        self.smpl_body = attach_pairof(smpl_body, cfg.smpl_cfg)
        if cfg.phase=='train' and not cfg.resume:
            weigt_name = cfg.smpl_cfg.get('pairof_pretrain')
            weight = torch.load(weigt_name)['state_dict']
            self.smpl_body.coap.load_state_dict(weight, strict=False)
            print(f'Load pretrained coap: {weigt_name}')
            if cfg.ignore_smpl_body == 'encoder':
                set_requires_grad(self.smpl_body.coap.part_encoder, False)
                print('freeze coap part encoder')
                if hasattr(self.smpl_body.coap, 'local_encoder'):
                    set_requires_grad(self.smpl_body.coap.local_encoder, False)
                    print('freeze coap local encoder')
            elif cfg.ignore_smpl_body == 'all':
                set_requires_grad(self.smpl_body, False)
                print('freeze coap')

        # color dict
        self.color_dict = nn.Parameter(
            torch.randn([self.smpl_body.coap.partitioner.sample_face_tensor.shape[0], cfg.rendering_network.code_dim]), requires_grad=True 
        )

        # non-rigid motion st positional encoding
        if not cfg.ignore_non_rigid_motions:
            self.get_non_rigid_embedder = \
                load_positional_embedder(cfg.non_rigid_embedder.module)

            # non-rigid motion MLP
            _, non_rigid_pos_embed_size = \
                self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
                                            cfg.non_rigid_motion_mlp.i_embed)
            self.non_rigid_mlp = \
                load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
                    pos_embed_size=non_rigid_pos_embed_size,
                    condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
                    mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
                    mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
                    skips=cfg.non_rigid_motion_mlp.skips)

        # color
        self.color_network = \
            load_render_mlp(cfg.rendering_network.module)(**cfg.rendering_network)

        # deform
        if not cfg.ignore_deform:
            self.center_template = torch.mm(self.smpl_body.J_regressor, self.smpl_body.v_template)[4]
            self.deformer = load_deform_mlp(cfg.deform_network.module)(verts=self.smpl_body.v_template[None], **cfg.deform_network)

        # pose decoder MLP
        if not cfg.ignore_pose_decoder:
            self.pose_decoder = \
                load_pose_decoder(cfg.pose_decoder.module)(
                    embedding_size=cfg.pose_decoder.embedding_size,
                    mlp_width=cfg.pose_decoder.mlp_width,
                    mlp_depth=cfg.pose_decoder.mlp_depth)
        
        # shadow_network
        if not cfg.ignore_shadow_network:
            self.shadow_network = load_shadow_network(cfg.shadow_network.module)(**cfg.shadow_network)
            sample_points = self.smpl_body.coap.partitioner.sample_points * cfg.smpl_cfg.scale
            if cfg.shadow_network.multires>0:
                get_embedder = load_positional_embedder(cfg.shadow_network.embedder)
                embed_fn, _ = get_embedder(cfg.shadow_network.multires)
                sample_points = embed_fn(sample_points)
            self.register_buffer('anchor_embed', sample_points)


    def deploy_mlps_to_secondary_gpus(self):
        pass

        return self


    def _query_mlp(
            self,
            pos_xyz,
            smpl_output,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        smpl_output=smpl_output,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        chunk=chunk)

        output = {}

        alpha_flat = result['alpha']
        feat_flat = result['feat']
        all_out_flat = result['all_out']
        output['alpha'] = torch.reshape(
                            alpha_flat, 
                            list(pos_xyz.shape[:-1]) + [alpha_flat.shape[-1]])
        output['feat'] = torch.reshape(
                            feat_flat, 
                            list(pos_xyz.shape[:-1]) + [feat_flat.shape[-1]])
        output['all_out'] = torch.reshape(
                            all_out_flat, 
                            list(pos_xyz.shape[:-1]) + [all_out_flat.shape[-1]])
        if not cfg.ignore_shadow_network:
            part_alpha = result['part_alpha']
            output['part_alpha'] = torch.reshape(
                            part_alpha, 
                            list(pos_xyz.shape[:-1]) + [part_alpha.shape[-1]])
            part_alpha = result['part_alpha']
            anchor_embed = result['anchor_embed']
            output['anchor_embed'] = torch.reshape(
                            anchor_embed, 
                            list(pos_xyz.shape[:-1]) + [anchor_embed.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            smpl_output,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            chunk):
        alpha_list = []
        feat_list = []
        part_alpha_list = []
        anchor_embed_list = []
        all_out_list = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end]

            if not cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem)
                )
                xyz = result['xyz']
            alpha, part_info = self.smpl_body.coap.query(xyz[None], smpl_output, ret_intermediate=True)
            alpha_list += [alpha[0, :, None]]
            all_out_list += [part_info['all_out'][0, :, None]]

            dists = part_info['gdists'][0]
            indices = part_info['gindices'][0]

            n_point, n_neighbor = indices.shape
            ws = (1. / dists).unsqueeze(-1)
            ws = ws / ws.sum(-2, keepdim=True)
            sel_latent = torch.gather(
                self.color_dict, 0, 
                indices.unsqueeze(-1).view(n_point*n_neighbor, -1).expand(-1, cfg.rendering_network.code_dim)
                ).view(n_point, n_neighbor, -1)
            sel_latent = (sel_latent * ws).sum(dim=-2)
            feat_list += [sel_latent, ]

            if not cfg.ignore_shadow_network:
                part_occ = part_info['soft_part_occupancy'][0].permute(1, 0).detach()
                sel_embed = torch.gather(
                    self.anchor_embed, 0, 
                    indices.unsqueeze(-1).view(n_point*n_neighbor, -1).expand(-1, self.anchor_embed.shape[-1])
                ).view(n_point, n_neighbor, -1)
                sel_embed = (sel_embed * ws).sum(dim=-2)
                part_alpha_list += [part_occ]
                anchor_embed_list += [sel_embed]

        output = {}
        output['alpha'] = torch.cat(alpha_list, dim=0).to(cfg.secondary_gpus[0])
        output['all_out'] = torch.cat(all_out_list, dim=0).to(cfg.secondary_gpus[0])
        output['feat'] = torch.cat(feat_list, dim=0).to(cfg.secondary_gpus[0])
        if not cfg.ignore_shadow_network:
            output['part_alpha'] = torch.cat(part_alpha_list, dim=0).to(cfg.secondary_gpus[0])
            output['anchor_embed'] = torch.cat(anchor_embed_list, dim=0).to(cfg.secondary_gpus[0])

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            smpl_output, dst_Th=None,
            non_rigid_pos_embed_fn=None,
            non_rigid_mlp_input=None,
            bgcolor=None,
            **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        if dst_Th is not None:
            pts -= dst_Th
        dirs = rays_d[:, None, :].expand(pts.shape)
        N_samples = pts.shape[1]

        # occ
        query_result = self._query_mlp(
                                pos_xyz=pts,
                                smpl_output=smpl_output,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        alpha = query_result['alpha'][:, :, 0]
        all_out = query_result['all_out'][:, :, 0]
        feat = query_result['feat']
        
        # color
        if cfg.rendering_network.mode == 'feat':
            sampled_color = self.color_network(None, None, None, feat.reshape(-1, feat.shape[-1])).reshape(N_rays, N_samples, 3)
        else:
            normed_dir = (dirs / dirs.norm(dim=-1, keepdim=True)).reshape(-1, 3)
            joints = smpl_output.joints[0]
            bones = []
            for p, c in enumerate(self.child):
                if c is None:
                    bones.append(joints[p])
                else:
                    bones.append(joints[c] - joints[p])
            bones = (torch.stack(bones).reshape(-1))[None].expand(N_rays*N_samples, -1)
            feat = feat.reshape(-1, feat.shape[-1])
            sampled_color = self.color_network(None, bones, normed_dir, feat).reshape(N_rays, N_samples, 3)

        if not cfg.ignore_shadow_network:
            part_alpha = torch.cummax(query_result['part_alpha'], dim=1).values.reshape(-1, query_result['part_alpha'].shape[-1])
            anchor_embed = query_result['anchor_embed'].reshape(-1, query_result['anchor_embed'].shape[-1])
            full_pose = smpl_output.full_pose.expand(part_alpha.shape[0], -1)
            sampled_shadow = self.shadow_network(torch.cat([part_alpha, anchor_embed, full_pose], -1)).reshape(N_rays, N_samples, -1)
            if not self.training:
                sampled_albedo = sampled_color.clone()
            if sampled_shadow.shape[-1] > 1:
                sampled_specular = sampled_shadow[..., 1:]
                sampled_shadow = sampled_shadow[..., :1]
                sampled_color = sampled_color * sampled_shadow + sampled_specular
            else:
                sampled_color = sampled_color * sampled_shadow
                sampled_specular = None

        # render
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = (sampled_color * weights[:, :, None]).sum(dim=1) # torch.Size([N_rays, 3])
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[..., None]) * (bgcolor[None, :]/255).to(rgb_map.device)

        res = {'rgb': rgb_map.to(cfg.primary_gpus[0]),  
                'alpha': acc_map.to(cfg.primary_gpus[0]),
                }
        if not self.training and not cfg.ignore_shadow_network:
            shadow_map = (sampled_shadow[..., 0] * weights).sum(dim=1)
            albedo_map = (sampled_albedo * weights[:, :, None]).sum(dim=1)
            albedo_map = albedo_map + (1.-acc_map[..., None]) * (bgcolor[None, :]/255).to(albedo_map.device)
            res.update({
                'shadow': shadow_map.to(cfg.primary_gpus[0]),
                'albedo': albedo_map.to(cfg.primary_gpus[0]),
                'weights': weights.to(cfg.primary_gpus[0]),
                'sampled_albedo': sampled_albedo.to(cfg.primary_gpus[0]),
                'sampled_shadow': sampled_shadow.to(cfg.primary_gpus[0])
                })
            if sampled_specular is not None:
                if sampled_specular.shape[-1]>1:
                    specular_map = (sampled_specular * weights[:, :, None]).sum(dim=1)
                    specular_map = specular_map + (1.-acc_map[..., None]) * (bgcolor[None, :]/255).to(specular_map.device)
                else:
                    specular_map = (sampled_specular[..., 0] * weights).sum(dim=1)
                res.update({
                    'specular': specular_map.to(cfg.primary_gpus[0]),
                    'sampled_specular': sampled_specular.to(cfg.primary_gpus[0])
                })
        elif 'iou3d' in list(cfg.train.lossweights.keys()):
            res.update({
                    'pred_occ': alpha.to(cfg.primary_gpus[0]),
                    'pts': pts.to(cfg.primary_gpus[0]),
                    'all_out': all_out.to(cfg.primary_gpus[0]),
                })

        return res

    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones -1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    @torch.no_grad()
    def sample_points_on_mesh(self, vertices, joints, full_pose):
        bone_trans = self.smpl_body.coap.compute_bone_trans(full_pose, joints)
        bbox_min, bbox_max = self.smpl_body.coap.get_bbox_bounds(vertices, bone_trans)  # (B, K, 1, 3) [can space]
        bbox_min = bbox_min.cpu()
        bbox_max = bbox_max.cpu()
        bone_trans = bone_trans.cpu()
        n_parts = bbox_max.shape[1]

        #### Sample points inside local boxes
        n_points_uniform = int(cfg.N_3d * 0.5)
        n_points_surface = cfg.N_3d - n_points_uniform

        bbox_size = (bbox_max - bbox_min).abs()*self.smpl_body.coap.bbox_padding - 1e-3  # (B,K,1,3)
        bbox_center = (bbox_min + bbox_max) * 0.5
        bb_min = (bbox_center - bbox_size*0.5)  # to account for padding
        
        uniform_points = bb_min + torch.rand((1, n_parts, n_points_uniform, 3)) * bbox_size  # [0,bs] (B,K,N,3)

        # project points to the posed space
        abs_transforms = torch.inverse(bone_trans)  # B,K,4,4
        uniform_points = (abs_transforms.reshape(1, n_parts, 1, 4, 4).repeat(1, 1, n_points_uniform, 1, 1) @ F.pad(uniform_points, [0, 1], "constant", 1.0).unsqueeze(-1))[..., :3, 0]

        #### Sample surface points
        face = self.smpl_body.coap.get_tight_face_tensor().cpu()
        meshes = Meshes(vertices.float().expand(n_parts, -1, -1).cpu(), face)
        surface_points = sample_points_from_meshes(meshes, num_samples=n_points_surface)
        surface_points += torch.from_numpy(np.random.normal(scale=0.001*cfg.smpl_cfg.scale, size=surface_points.shape))
        surface_points = surface_points.reshape((1, n_parts, -1, 3))

        points = torch.cat((uniform_points, surface_points), dim=-2).float()  # B,K,n_points,3

        #### Check occupancy
        points = points.reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices.squeeze().cpu().numpy(), self.smpl_body.faces_seal, process=False)
        gt_occ = check_mesh_contains(mesh, points.numpy()).astype(np.float32)

        # rgb = np.ones_like(points.numpy())
        # to_save = np.concatenate([points.numpy(), rgb * 255], axis=-1)
        # np.savetxt(f'/mnt/user/chenxingyu/jupyter/train2.ply',
        #             to_save,
        #             fmt='%.6f %.6f %.6f %d %d %d',
        #             comments='',
        #             header=(
        #                 'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
        #                 points.shape[0])
        #             )
        # import matplotlib.pyplot as plt
        # from coaphand.coap.coap_partpoint import COAPBodyModel
        # color = COAPBodyModel.get_part_colors()
        # points = points.numpy()
        # n_part = 16
        # points = points.reshape(n_part, 256, 3)
        # gt_occ = gt_occ.reshape(n_part, 256, 1)
        # gt_occ[gt_occ==0] = 0.5

        # fig = plt.figure(figsize=(8, 8))
        # ax = plt.subplot(1, 1, 1, projection='3d')
        # for i in range(n_part):
        #     c = color[i:i+1].repeat(256, axis=0)
        #     a = gt_occ[i]
        #     ca = np.concatenate([c, a], axis=1)
        #     ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2], c=ca)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax.set_xlim([-1.2, 1.2])
        # ax.set_ylim([-1.2, 1.2])
        # ax.set_zlim([-1.2, 1.2])

        # plt.subplots_adjust(left=0., right=0.95, top=0.97, bottom=0.03, wspace=0., hspace=0.)
        # plt.savefig(f'tmp/train.png')

        return points.to(cfg.primary_gpus[0]), torch.from_numpy(gt_occ).to(cfg.primary_gpus[0])

    def forward(self,
                rays, 
                dst_Rs, dst_Ts,
                dst_posevec, dst_shape, dst_global_orient, dst_Th=None,
                near=None, far=None,
                iter_val=1e7,
                **kwargs):

        dst_global_orient=dst_global_orient[None, ...]
        dst_shape=dst_shape[None, ...] * 0
        dst_posevec=dst_posevec[None, ...]
        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        if dst_Th is not None:
            dst_Th = dst_Th[None]

        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0) and not cfg.ignore_pose_decoder:
            pose_out = self.pose_decoder(dst_posevec)
            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out.get('Ts', None)
            
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(
                                        dst_Rs_no_root, 
                                        refined_Rs)
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts

        if not cfg.ignore_deform:
            offsets = self.deformer(dst_posevec).to(cfg.primary_gpus[0])
            if cfg.deform_network.get('normal_frame'):
                normals = vertex_normals(self.smpl_body.v_template[None], self.smpl_body.faces_tensor[None])
                normal_coord_sys = get_normal_coord_system(normals.view(-1, 3)).view(1, self.smpl_body.v_template.shape[0], 3, 3)
                offsets = torch.matmul(normal_coord_sys.permute(0, 1, 3, 2), offsets.unsqueeze(-1)).squeeze(-1)            
            shaped_verts = self.smpl_body.v_template[None] + offsets
            center = torch.mm(self.smpl_body.J_regressor, shaped_verts[0])[4]
            shaped_verts = shaped_verts - center[None, None] + self.center_template[None, None].to(shaped_verts.device)
        else:
            shaped_verts = None

        smpl_output = self.smpl_body(
            dst_shape, 
            dst_global_orient, 
            dst_posevec, 
            return_verts=True, 
            return_full_pose=True,
            shaped_verts=shaped_verts,
            )
        kwargs.update({
                'smpl_output': smpl_output,
                'dst_Th': dst_Th
            })

        if not cfg.ignore_non_rigid_motions:
            non_rigid_pos_embed_fn, _ = \
                self.get_non_rigid_embedder(
                    multires=cfg.non_rigid_motion_mlp.multires,                         
                    is_identity=cfg.non_rigid_motion_mlp.i_embed,
                    iter_val=iter_val,)

            if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
                non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
            else:
                non_rigid_mlp_input = dst_posevec

            kwargs.update({
                "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
                "non_rigid_mlp_input": non_rigid_mlp_input
            })
            

        rays_o, rays_d = rays
        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        all_ret.update({'smpl_output': smpl_output, 'shaped_verts': shaped_verts,})

        if self.training and cfg.get('N_3d', 0)>0:
            with torch.no_grad():
                pts3d, gt_occ = self.sample_points_on_mesh(smpl_output.vertices.detach(), smpl_output.joints.detach(), smpl_output.full_pose)
            occ, part_info = self.smpl_body.coap.query(pts3d[None], smpl_output, ret_intermediate=True)
            all_ret.update({'pred_occ': occ[0], 'all_out': part_info['all_out'][0], 'gt_occ': gt_occ})

        return all_ret
