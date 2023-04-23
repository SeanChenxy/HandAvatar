import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
import cv2
from handavatar.core.utils.augm_util import process_bbox, augmentation
from handavatar.configs import cfg
import torch
from handavatar.core.utils.camera_util import apply_global_tfm_to_camera
from pycocotools.coco import COCO
import json
import smplx
from handavatar.core.utils.math_util import convert

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area

class Dataset(torch.utils.data.Dataset):

    @torch.no_grad()
    def __init__(self, subject):
        # mano
        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        self.mano = smplx.create(**cfg.smpl_cfg)
        self.handtype = ('left', 'right')[cfg.smpl_cfg.is_rhand]
        if self.handtype=='left':
            self.mano.shapedirs[:,0,:] *= -1

        # annotation
        self.image_dir = os.path.join('data/InterHand/5', 'InterHand2.6M_5fps_batch1/images')
        anno_name = os.path.join(self.image_dir.replace('images', 'preprocess'), subject, f'anno_{self.handtype}_for_mask.pkl')
        print('Load annotation', anno_name)
        if not os.path.exists(anno_name):
            print('Preprocessing ...')
            self.preprocess('data/InterHand/5', anno_name)
        with open(anno_name, 'rb') as f:
            self.cameras, self.mesh_infos, self.bbox, self.framelist = pickle.load(f)            

    def preprocess(self, dataset_path, anno_name):

        th_hands_mean_right = np.array([0.1117, -0.0429, 0.4164, 0.1088, 0.0660, 0.7562, -0.0964, 0.0909,
                                                 0.1885, -0.1181, -0.0509, 0.5296, -0.1437, -0.0552, 0.7049, -0.0192,
                                                 0.0923, 0.3379, -0.4570, 0.1963, 0.6255, -0.2147, 0.0660, 0.5069,
                                                 -0.3697, 0.0603, 0.0795, -0.1419, 0.0859, 0.6355, -0.3033, 0.0579,
                                                 0.6314, -0.1761, 0.1321, 0.3734, 0.8510, -0.2769, 0.0915, -0.4998,
                                                 -0.0266, -0.0529, 0.5356, -0.0460, 0.2774])
        th_hands_mean_left = th_hands_mean_right.copy().reshape(-1, 3)
        th_hands_mean_left[:, 1:] *= -1
        th_hands_mean_left = th_hands_mean_left.reshape(-1)

        phase = cfg.subject.split('/')[0]
        dir_name = '/'.join(cfg.subject.split('/')[1:])

        self.annot_path = os.path.join(dataset_path, 'annotations')
        print("Load annotation from  " + os.path.join(self.annot_path, phase))
        db = COCO(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_data.json'))
        with open(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_camera.json')) as f:
            cameras = json.load(f)
        with open(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_joint_3d.json')) as f:
            joints_interhand = json.load(f)
        with open(os.path.join(self.annot_path, phase, 'InterHand2.6M_' + phase + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)

        print()
        self.cameras = {}
        self.mesh_infos = {}
        self.bbox = {}
        self.framelist = []

        for i, aid in enumerate(db.anns.keys()):
            if i%5000==0:
                print(i, len(db.anns.keys()))

            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            if dir_name+'/' not in img['file_name']:
                continue
            capture_id = img['capture']
            cam = img['camera']
            frame_idx = img['frame_idx']
            image_name = img['file_name']
            hand_type = ann['hand_type']
            try:
                mano_param = mano_params[str(capture_id)][str(frame_idx)][self.handtype]
            except:
                print('cannot read mano params', image_name)
                continue
            if hand_type != self.handtype or mano_param is None:
                # print(f'{i}, Discard {image_name}, {hand_type} is not agree with {self.handtype}')
                continue
            
            # bbox
            img_width, img_height = img['width'], img['height']
            bbox = np.array(ann['bbox'], dtype=np.float32) # x,y,w,h
            if bbox[0]<10 or bbox[1]<10 or max(bbox[2], bbox[3])<80 or bbox[0]+bbox[2]>img_width-10 or bbox[1]+bbox[3]>img_height-10:
                continue
            bbox = process_bbox(bbox, img_width, img_height)

            # frame
            img_path = os.path.join(self.image_dir, f'{phase}/{image_name}')
            img = cv2.imread(img_path)
            if img.max() < 20:
                continue

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

            root_index = (41, 20)[self.handtype=='right']
            self.mesh_infos[f'{phase}/{image_name}'] = {
                'Rh': np.zeros(3),
                'Th': joint_cam[root_index] - joints[0],
                'poses': poses.reshape(-1),
                'shape': betas,
                'joints': joints,
                'tpose_joints': tpose_joints,
            }

        os.makedirs(os.path.dirname(anno_name), exist_ok=True)
        with open(anno_name, 'wb') as f:
            pickle.dump([self.cameras, self.mesh_infos, self.bbox, self.framelist], f)

    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'shape': self.mesh_infos[frame_name]['shape'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32'),
        }

    @torch.no_grad()
    def mask_from_mano(self, idx, filtering=True, vis=False):
        frame_name = self.framelist[idx]
        save_name = os.path.join(self.image_dir, frame_name).replace('images', 'masks_removeblack').replace('.jpg', '.png')

        dst_skel_info = self.query_dst_skeleton(frame_name)
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        E = self.cameras[frame_name]['extrinsics']

        poses = dst_skel_info['poses']
        betas = dst_skel_info['shape']
        Rh = dst_skel_info['Rh']
        Th = dst_skel_info['Th']

        E = apply_global_tfm_to_camera(
            E=E,
            Rh=dst_skel_info['Rh'],
            Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]

        posed_res = self.mano(
                torch.from_numpy(betas)[None].float(), 
                torch.from_numpy(poses[:3])[None].float(), 
                torch.from_numpy(poses[3:])[None].float(),
                return_verts=True)
        joints = posed_res.joints[0].numpy()
        verts = posed_res.vertices[0].numpy()
        verts = np.dot(R, verts.T).T + T
        verts_img = np.dot(K, verts.T).T
        verts_img[:, :2] = np.round(verts_img[:, :2] / verts_img[:, 2:3])
        verts_img = verts_img.astype(np.int32)

        img = cv2.imread(os.path.join(self.image_dir, frame_name))
        
        mask = np.zeros_like(img)
        for f in self.mano.faces:
            triangle = np.array([[verts_img[f[0]][0], verts_img[f[0]][1]], [verts_img[f[1]][0], verts_img[f[1]][1]], [verts_img[f[2]][0], verts_img[f[2]][1]]])
            cv2.fillConvexPoly(mask, triangle, (255,255,255))
        
        if filtering:
            if mask.max()<20:
                print(i, frame_name, 'mask is all black')
                return
            
            mask_bool = mask[..., 0]==255
            sel_img = img[mask_bool].mean(axis=-1)
            if sel_img.max()<20:
                print(i, frame_name, 'sel_img is all black')
                return

            sel_img = np.bitwise_and(sel_img>10, sel_img<200)
            mask_bool[mask_bool] = sel_img.astype('int32')
            mask = mask * mask_bool[..., None]

            contours, _ = cv2.findContours(mask[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(contours)
            contours.sort(key=cnt_area, reverse=True)
            poly = contours[0].transpose(1, 0, 2).astype(np.int32)
            poly_mask = np.zeros_like(img)
            poly_mask = cv2.fillPoly(poly_mask, poly, (1,1,1))
            mask = mask * poly_mask

        if vis:
            mask_red = np.concatenate([np.zeros(list(mask[..., 0].shape) + [2]), mask[..., 0:1]], 2).astype(np.uint8)
            img_mask = cv2.addWeighted(img.astype(np.uint8), 1, mask_red, 0.5, 0)
            cv2.imwrite(f'tmp/{idx}.png', img_mask)
        else:
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            cv2.imwrite(save_name, mask)

    def __len__(self):
        return len(self.framelist)


if __name__ == '__main__':
    subject = 'test/Capture0'
    print(subject)
    dataset = Dataset(subject)
    for i in range(0, len(dataset)):
        if i % 1000==0:
            print(f'{i} / {len(dataset)}')
        dataset.mask_from_mano(i, filtering=True, vis=False)
