import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import pickle

import numpy as np
import cv2
import torch

from handavatar.core.utils.image_util import load_image
from handavatar.core.utils.hand_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from handavatar.core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox, \
    cam2pixel

from handavatar.core.utils.math_util import convert
from handavatar.core.utils.hand_util import MANOHand, INTERHAND2MANO
from handavatar.core.utils.augm_util import process_bbox, augmentation, trans_point2d

import json
from pycocotools.coco import COCO
from handavatar.configs import cfg
import smplx


class Dataset(torch.utils.data.Dataset):

    @torch.no_grad()
    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            subject=None,
            **kwargs):

        print('[Dataset Path]', dataset_path)

        # MANO
        self.mano = smplx.create(**cfg.smpl_cfg)
        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype=='left':
            self.mano.shapedirs[:,0,:] *= -1
        
        # annotation
        self.phase = kwargs.get('data_type', 'train')
        if subject is None:
            if self.phase == 'train':
                subject = cfg.subject
            else:
                subject = cfg[kwargs['data_type']].subject
        if isinstance(subject, list):
            subject = subject[0]
        self.image_dir = os.path.join(dataset_path, f'InterHand2.6M_{cfg.interhand.fps}fps_batch1/images')
        anno_name = os.path.join(self.image_dir.replace('images', 'preprocess'), subject, 'anno_cam.pkl')
        print('Load annotation', anno_name)
        if not os.path.exists(anno_name):
            print('Preprocessing ...')
            self.preprocess(dataset_path, subject, anno_name, self.phase)
        with open(anno_name, 'rb') as f:
            self.cameras, self.mesh_infos, self.bbox, self.framelist = pickle.load(f)

        # canonical
        mano_res = self.mano(
            torch.zeros(1, 10).float(), 
            torch.zeros(1, 3).float(), 
            torch.zeros(1, 45).float(),
            return_verts=True)
        self.canonical_bbox = mano_res.joints[0].numpy()
        self.canonical_verts = mano_res.vertices[0].numpy()
        self.canonical_bbox = self.skeleton_to_bbox(self.canonical_verts)

        # post process
        self.framelist = self.framelist[::skip]
        exclude = cfg[kwargs['data_type']].get('exclude_idx', None)
        if isinstance(exclude, list):
            sel_idx = list(range(len(self.framelist)))
            self.framelist = [self.framelist[i] for i in sel_idx if i not in exclude]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]
        print(f' -- Total Frames: {self.get_total_frames()}')
        self.keyfilter = keyfilter
        self.bgcolor = bgcolor
        self.ray_shoot_mode = ray_shoot_mode

    @torch.no_grad()
    def preprocess(self, dataset_path, subject, anno_name, data_type, with_mask=True):

        th_hands_mean_right = np.array([0.1117, -0.0429, 0.4164, 0.1088, 0.0660, 0.7562, -0.0964, 0.0909,
                                        0.1885, -0.1181, -0.0509, 0.5296, -0.1437, -0.0552, 0.7049, -0.0192,
                                        0.0923, 0.3379, -0.4570, 0.1963, 0.6255, -0.2147, 0.0660, 0.5069,
                                        -0.3697, 0.0603, 0.0795, -0.1419, 0.0859, 0.6355, -0.3033, 0.0579,
                                        0.6314, -0.1761, 0.1321, 0.3734, 0.8510, -0.2769, 0.0915, -0.4998,
                                        -0.0266, -0.0529, 0.5356, -0.0460, 0.2774])
        th_hands_mean_left = th_hands_mean_right.copy().reshape(-1, 3)
        th_hands_mean_left[:, 1:] *= -1
        th_hands_mean_left = th_hands_mean_left.reshape(-1)

        phase = subject.split('/')[0]
        dir_name = '/'.join(subject.split('/')[1:])

        self.annot_path = os.path.join(dataset_path, 'annotations')
        print("Load annotation from  " + os.path.join(self.annot_path, phase))
        db = COCO(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_data.json'))
        with open(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_camera.json')) as f:
            cameras = json.load(f)
        with open(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_joint_3d.json')) as f:
            joints_interhand = json.load(f)
        with open(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)

        self.cameras = {}
        self.mesh_infos = {}
        self.bbox = {}
        self.framelist = []

        for i, aid in enumerate(db.anns.keys()):
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            if dir_name+'/' not in img['file_name']:
                continue
            if i%5000==0:
                print(i)
            capture_id = img['capture']
            cam = img['camera']
            frame_idx = img['frame_idx']
            image_name = img['file_name']
            hand_type = ann['hand_type']
            joint_valid = np.array(ann['joint_valid'])
            try:
                mano_param = mano_params[str(capture_id)][str(frame_idx)][self.handtype]
            except:
                print('cannot read mano params', image_name)
                continue
            if hand_type != self.handtype or mano_param is None:
                print(f'{i}, Discard {image_name}, {hand_type} is not agree with {self.handtype}')
                continue

                # bbox
            img_width, img_height = img['width'], img['height']
            bbox = np.array(ann['bbox'], dtype=np.float32) # x,y,w,h
            if data_type != 'infer':
                if bbox[0]<10 or bbox[1]<10 or max(bbox[2], bbox[3])<80 or bbox[0]+bbox[2]>img_width-10 or bbox[1]+bbox[3]>img_height-10:
                    # print(f'{i}, Discard {image_name}, bbox is too biased/small: {bbox.tolist()}')
                    continue

                # frame
                img_path = os.path.join(self.image_dir, f'{phase}/{image_name}')
                img = cv2.imread(img_path)
                if img.max() < 20:
                    # print(f'{i}, Discard {image_name}, RGB is too dark: {img.max()}')
                    continue
                if np.allclose(img[..., 0], img[..., 1], atol=1) or np.allclose(img[..., 2], img[..., 1], atol=1) or np.allclose(img[..., 0], img[..., 2], atol=1):
                    # print(f'{i}, Discard {image_name}, Gray scale')
                    continue

                # mask
                mask_path = img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png')
                if not os.path.exists(mask_path):
                    # print(f'{i}, Discard {image_name}, w/o mask')
                    continue
                mask = cv2.imread(img_path.replace('images', 'masks_removeblack').replace('.jpg', '.png'))
                mask_sum = mask[..., 0].astype('bool').sum()
                if mask.max() < 255 or mask_sum < 3000:
                    # print(f'{i}, Discard {image_name}, mask is too dark: {mask.max()}, {mask_sum}')
                    continue
                
                mask_bool = mask[..., 0]==255
                sel_img = img[mask_bool].mean(axis=-1)
                if sel_img.max()<20:
                    print(i, frame_name, 'sel_img is too dark')
                    continue

            bbox = process_bbox(bbox, img_width, img_height)
            self.bbox[f'{phase}/{image_name}'] = bbox
            self.framelist.append(f'{phase}/{image_name}')

            # camera
            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            E = np.eye(4)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            K = np.eye(3)
            K[[0, 1], [0, 1]] = focal
            K[[0, 1], [2, 2]] = princpt
            self.cameras[f'{phase}/{image_name}'] = {
                'intrinsics': K,
                'extrinsics': E,
                'distortions': np.zeros(5)
            }

            # mesh
            poses = np.array(mano_param['pose'])
            betas = np.array(mano_param['shape'])
            Rh = poses[:3].copy()
            Rh_mat = np.dot( camrot, convert(Rh, 'axangle', 'rotmat') )
            Rh = convert(Rh_mat, 'rotmat',  'axangle')
            poses[:3] = Rh
            poses[3:] += (th_hands_mean_left, th_hands_mean_right)[self.handtype=='right']
            tres = self.mano(
                torch.from_numpy(betas)[None].float(), 
                torch.zeros(1, 3).float(), 
                torch.zeros(1, 45).float(),
                return_verts=True)
            tpose_joints = tres.joints[0].numpy()
            tverts = tres.vertices[0].numpy()

            posed_res = self.mano(
                torch.from_numpy(betas)[None].float(), 
                torch.from_numpy(poses)[None, :3].float(), 
                torch.from_numpy(poses)[None, 3:].float(),
                return_verts=True)
            joints = posed_res.joints[0].numpy()
            verts = posed_res.vertices[0].numpy()

            joint_world = np.array(joints_interhand[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32) / 1000 * cfg.smpl_cfg.scale
            joint_cam = np.dot(camrot, joint_world.T).T - np.dot(camrot, campos) / 1000 * cfg.smpl_cfg.scale
            joint_cam = joint_cam[:21][INTERHAND2MANO] if self.handtype=='right' else joint_cam[21:][INTERHAND2MANO]
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2].astype('int32')
            joint_valid = joint_valid[:21][INTERHAND2MANO] if self.handtype=='right' else joint_valid[21:][INTERHAND2MANO]

            self.mesh_infos[f'{phase}/{image_name}'] = {
                'Rh': np.zeros(3),
                'Th': joint_cam[0] - joints[0],
                'poses': poses.reshape(-1),
                'shape': betas,
                'joints': joints,
                'tpose_joints': tpose_joints,
                'bbox': self.skeleton_to_bbox(verts),
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid
            }
        
        os.makedirs(os.path.dirname(anno_name), exist_ok=True)
        with open(anno_name, 'wb') as f:
            pickle.dump([self.cameras, self.mesh_infos, self.bbox, self.framelist], f)

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }


    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'shape': self.mesh_infos[frame_name]['shape'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32'),
            'joint_img': self.mesh_infos[frame_name]['joint_img'].astype('int32'),
            'joint_cam': self.mesh_infos[frame_name]['joint_cam'].astype('float32'),
            'joint_valid': self.mesh_infos[frame_name]['joint_valid'].astype('float32')
        }

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, ray_alpha, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        ray_alpha = ray_alpha[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, ray_alpha, near, far
    
    def get_patch_ray_indices(
            self, 
            N_patch, 
            ray_mask, 
            subject_mask, 
            bbox_mask,
            patch_size, 
            H, W):

        assert subject_mask.dtype == np.bool_
        assert bbox_mask.dtype == np.bool_

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]
        for _ in range(N_patch):
            # let p = cfg.patch.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max = \
                self._get_patch_ray_indices(ray_mask, candidate_mask, 
                                            patch_size, H, W)

            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices)

            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)
            
            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0)
        }
        patch_div_indices = np.array(patch_div_indices)

        return select_inds, patch_info, patch_div_indices


    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool_
        assert candidate_mask.dtype == np.bool_

        valid_ys, valid_xs = np.where(candidate_mask)

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
        half_patch_size = patch_size // 2
        x_min = np.clip(a=center_x-half_patch_size, 
                        a_min=0, 
                        a_max=W-patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(a=center_y-half_patch_size,
                        a_min=0,
                        a_max=H-patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]
        
        inter_mask = inter_mask.reshape(H, W)

        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
                np.array([x_min, y_min]), np.array([x_max, y_max])
    
    def load_image(self, frame_name, bg_color, use_mask=True):
        imagepath = os.path.join(self.image_dir, frame_name)
        orig_img = np.array(load_image(imagepath))

        if use_mask:
            maskpath = imagepath.replace('images', 'masks_removeblack').replace('.jpg', '.png')
            alpha_mask = np.array(load_image(maskpath))
        else:
            alpha_mask = np.ones_like(orig_img) * 255
        
        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        img = alpha_mask / 255. * orig_img + (1.0 - alpha_mask / 255.) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None,
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask


    def get_total_frames(self):
        return len(self.framelist)

    def sample_patch_rays(self, img, alpha, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, ray_img, ray_alpha, near, far):

        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=cfg.patch.N_patches, 
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=cfg.patch.size, 
                H=H, W=W)

        rays_o, rays_d, ray_img, ray_alpha, near, far = self.select_rays(
            select_inds, rays_o, rays_d, ray_img, ray_alpha, near, far)
        
        targets = []
        targets_alpha = []
        for i in range(cfg.patch.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(img[y_min:y_max, x_min:x_max])
            targets_alpha.append(alpha[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)
        target_alpha_patches = np.stack(targets_alpha, axis=0)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, ray_img, ray_alpha, near, far, \
                target_patches, target_alpha_patches, patch_masks, patch_div_indices

    def __len__(self):
        return self.get_total_frames()

    def __getitem__(self, idx):
        frame_name = self.framelist[idx]
        results = {
            'frame_name': frame_name
        }

        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        img, alpha = self.load_image(frame_name, bgcolor, use_mask=(True, False)[self.phase=='infer'])
        bbox = self.bbox[frame_name]
        img, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, alpha = augmentation(img, self.bbox[frame_name], 'eval',
                                                                                     exclude_flip=True,
                                                                                     input_img_shape=(256, 256), mask=alpha,
                                                                                     base_scale=1.3,
                                                                                     scale_factor=0.2,
                                                                                     rot_factor=0,
                                                                                     shift_wh=[bbox[2], bbox[3]],
                                                                                     gaussian_std=3,
                                                                                     bordervalue=bgcolor.tolist())

        # cv2.imwrite('test.png', img.astype('uint8'))
        # cv2.waitKey(0)

        img = (img / 255.).astype('float32')
        alpha = alpha.astype('float32')
        results['target_alpha_img'] = alpha[..., 0]

        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_shape = dst_skel_info['shape']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_cam_joints = dst_skel_info['joint_cam']
        dst_valid_joints = dst_skel_info['joint_valid']

        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2, 2] = trans_point2d(K[:2, 2], img2bb_trans)
        K[[0, 1], [0, 1]] = K[[0, 1], [0, 1]] * 256 / (bbox[2]*aug_param[1])

        E = self.cameras[frame_name]['extrinsics']

        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]
        results.update({
            'cam_K': K,
            'cam_R': R,
            'cam_T': T,
        })
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)
        ray_alpha = alpha.reshape(-1, 3) 

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]
        ray_alpha = ray_alpha[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        if self.ray_shoot_mode == 'image':
            pass
        elif self.ray_shoot_mode == 'patch':
            rays_o, rays_d, ray_img, ray_alpha, near, far, \
            target_patches, target_alpha_patches, patch_masks, patch_div_indices = \
                self.sample_patch_rays(img=img, alpha=alpha, H=H, W=W,
                                       subject_mask=alpha[:, :, 0] > 0.,
                                       bbox_mask=ray_mask.reshape(H, W),
                                       ray_mask=ray_mask,
                                       rays_o=rays_o, 
                                       rays_d=rays_d, 
                                       ray_img=ray_img, 
                                       ray_alpha=ray_alpha,
                                       near=near, 
                                       far=far)
        else:
            assert False, f"Ivalid Ray Shoot Mode: {self.ray_shoot_mode}"
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})

            if self.ray_shoot_mode == 'patch':
                results.update({
                    'patch_div_indices': patch_div_indices,
                    'patch_masks': patch_masks,
                    'target_patches': target_patches,
                    'target_alpha_patches': target_alpha_patches[..., 0],
                    })

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img
            results['target_alpha'] = ray_alpha[..., 0]
            results['target_img'] = img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                dst_poses, dst_tpose_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
            })

        if 'dst_posevec_69' in self.keyfilter:
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
                'dst_shape': dst_shape,
                'dst_global_orient': dst_poses[:3],
                'dst_cam_joints': dst_cam_joints,
                'dst_valid_joints': dst_valid_joints[:, 0]
            })

        return results

    @torch.no_grad()
    def vis(self, data, idx):
        import matplotlib.pyplot as plt
        from handavatar.third_parties.smpl.smpl_numpy import SMPL
        import cv2
        import vctoolkit as vc

        frame_name = self.framelist[idx]
        print(frame_name)

        fig = plt.figure()
        sub = 7
        ax = plt.subplot(1, sub, 1)
        img = np.zeros([data['img_height'], data['img_width'], 3], dtype=np.float32)
        ro = data['rays'][0]
        rd = data['rays'][1]
        world_p = ro + rd
        cam_p = np.dot(data['cam_R'], world_p.T).T + data['cam_T']
        img_p = np.round( np.dot(data['cam_K'], cam_p.T).T[:, :2] ).astype(np.int32)
        if 'target_rgbs' in data:
            for i in range(data['target_rgbs'].shape[0]):
                img[img_p[i, 1], img_p[i, 0]] = data['target_rgbs'][i]
        else:
            patch = data['target_patches'].reshape(-1, 3)
            for i in range(img_p.shape[0]):
                img[img_p[i, 1], img_p[i, 0]] = patch[i]
        img = (img * 255).astype(np.uint8)
        ax.imshow(img)
        ax.axis('off')

        ax = plt.subplot(1, sub, 2)
        alpha = np.zeros([data['img_height'], data['img_width']], dtype=np.float32) + 0.5
        if 'target_alpha' in data:
            for i in range(data['target_alpha'].shape[0]):
                alpha[img_p[i, 1], img_p[i, 0]] = data['target_alpha'][i]
        else:
            patch = data['target_alpha_patches'].reshape(-1)
            for i in range(img_p.shape[0]):
                alpha[img_p[i, 1], img_p[i, 0]] = patch[i]
        alpha = (alpha * 255).astype(np.uint8)
        ax.imshow(alpha)
        ax.axis('off')

        ax = plt.subplot(1, sub, 3)
        ray_mask = data['ray_mask'].reshape(data['img_height'], data['img_width']).astype(np.uint8)
        ax.imshow(ray_mask)
        ax.axis('off')

        ax = plt.subplot(1, sub, 4)
        posed_res = self.mano(
                torch.from_numpy(data['dst_shape'])[None].float(), 
                torch.from_numpy(data['dst_global_orient'])[None].float(), 
                torch.from_numpy(data['dst_posevec'])[None].float(),
                return_verts=True)
        joints = posed_res.joints[0].numpy()
        verts = posed_res.vertices[0].numpy()
        cam_v = np.dot(data['cam_R'], verts.T).T + data['cam_T']
        img_v = np.dot(data['cam_K'], cam_v.T).T
        img_v[:, :2] = np.round(img_v[:, :2]/img_v[:, 2:3])
        img_v = img_v.astype(np.int32)
        img_vis_v = img.copy()
        for i in range(img_v.shape[0]):
            cv2.circle(img_vis_v, (img_v[i, 0], img_v[i, 1]), 2, (255, 0, 0), -1)
        ax.imshow(img_vis_v)
        ax.axis('off')

        ax = plt.subplot(1, sub, 5)
        cam_j = np.dot(data['cam_R'], joints.T).T + data['cam_T']
        img_j = np.dot(data['cam_K'], cam_j.T).T
        img_j[:, :2] = np.round(img_j[:, :2]/img_j[:, 2:3])
        img_j = img_j[:, :2].astype(np.int32)
        img_vis_j = vc.render_bones_from_uv(np.flip(img_j, axis=-1).copy(), img.copy(), MANOHand)
        ax.imshow(img_vis_j)
        ax.axis('off')

        ax = plt.subplot(1, sub, 6)
        cam_c = data['dst_cam_joints']
        img_c = np.dot(data['cam_K'], cam_c.T).T
        img_c[:, :2] = np.round(img_c[:, :2]/img_c[:, 2:3])
        img_c = img_c[:, :2].astype(np.int32)
        img_vis_c = vc.render_bones_from_uv(np.flip(img_c, axis=-1).copy(), img.copy(), MANOHand)
        ax.imshow(img_vis_c)
        ax.axis('off')

        valid = data['dst_valid_joints']
        valid_mask = valid == 1

        ax = plt.subplot(1, sub, 7)
        world_p = np.array([[0, 0, 0],
                            [0.1*cfg.smpl_cfg.scale, 0, 0],
                            [0, 0.1*cfg.smpl_cfg.scale, 0],
                            [0, 0, 0.1*cfg.smpl_cfg.scale]])
        cam_o = np.dot(data['cam_R'], world_p.T).T + data['cam_T']
        img_o = np.dot(data['cam_K'], cam_o.T).T
        img_o[:, :2] = np.round(img_o[:, :2] / img_o[:, 2:3])
        img_o = img_o.astype(np.int32)
        img_vis_o = img.copy()
        cv2.line(img_vis_o, img_o[0][:2], img_o[1][:2], (255, 0, 0), 2)
        cv2.line(img_vis_o, img_o[0][:2], img_o[2][:2], (0, 255, 0), 2)
        cv2.line(img_vis_o, img_o[0][:2], img_o[3][:2], (0, 255, 255), 2)
        ax.imshow(img_vis_o)
        ax.axis('off')
        
        plt.savefig(f'tmp/{idx}.png')
        plt.close()


if __name__ == '__main__':

    from handavatar.core.data.dataset_args import DatasetArgs
    from handavatar.configs import cfg, make_cfg, args
    args.cfg = 'handavatar/configs/interhand/test_cap0.yaml'
    cfg = make_cfg(args)

    args = DatasetArgs.get('interhand_train')
    args['data_type'] = 'train'
    dataset = Dataset(**args)
    for i in range(0, len(dataset), len(dataset)//10):
        print(f'{i} / {len(dataset)}')
        data = dataset.__getitem__(i)
        dataset.vis(data, i)
