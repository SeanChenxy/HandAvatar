import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.utils.data as data
import numpy as np
from utils.fh_utils import load_db_annotation, read_mesh, read_img, read_img_abs, read_mask_woclip, projectPoints
from utils.vis import base_transform, inv_based_tranmsform, cnt_area
import cv2
from termcolor import cprint
from utils.preprocessing import augmentation, augmentation_2d
from mobhand.tools.kinematics import MPIIHandJoints
from mobhand.tools.joint_order import MANO2MPII
from mobhand.models.loss import SilhouetteLoss, ProjectionLoss
import vctoolkit as vc


class FreiHAND(data.Dataset):

    def __init__(self):
        super(FreiHAND, self).__init__()
        self.phase = 'training'
        self.db_data_anno = tuple(load_db_annotation('data/FreiHAND/data', set_name=self.phase))
        self.data_size = 224
        cprint('Loaded FreiHand {} {} samples'.format(self.phase, str(len(self.db_data_anno))), 'red')

    def __getitem__(self, idx):
        return self.get_training_sample(idx)


    def get_training_sample(self, idx):
        # read
        img = read_img_abs(idx, 'data/FreiHAND/data', 'training')
        vert = read_mesh(idx, 'data/FreiHAND/data').x.numpy()
        mask = read_mask_woclip(idx, 'data/FreiHAND/data', 'training')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cnt_area, reverse=True)
        bbox = cv2.boundingRect(contours[0])
        center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
        w, h = bbox[2], bbox[3]
        bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]
        K, mano, joint_cam = self.db_data_anno[idx]
        K, joint_cam, mano = np.array(K), np.array(joint_cam), np.array(mano)
        joint_img = projectPoints(joint_cam, K)
        princpt = K[0:2, 2].astype(np.float32)
        focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)

        # augmentation
        roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(img, bbox, 'eval',
                                                                                        exclude_flip=True,
                                                                                        input_img_shape=(self.data_size, self.data_size),
                                                                                        mask=mask,
                                                                                        base_scale=1.3,
                                                                                        scale_factor=0.2,
                                                                                        rot_factor=0,
                                                                                        shift_wh=[bbox[2], bbox[3]],
                                                                                        gaussian_std=3)

        roi = base_transform(roi, self.data_size, mean=0.5, std=0.5)
        # img = inv_based_tranmsform(roi)
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        roi = torch.from_numpy(roi).float()
        mask = torch.from_numpy(mask).float()
        bb2img_trans = torch.from_numpy(bb2img_trans).float()

        # joints
        joint_img, princpt = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)

        # K
        focal = focal * roi.size(1) / (bbox[2]*aug_param[1])
        calib = np.eye(4)
        calib[0, 0] = focal[0]
        calib[1, 1] = focal[1]
        calib[:2, 2:3] = princpt[:, None]
        calib = torch.from_numpy(calib).float()

        # postprocess root and joint_cam
        root = joint_cam[0].copy()
        joint_cam -= root
        vert -= root
        root = torch.from_numpy(root).float()
        joint_cam = torch.from_numpy(joint_cam).float()
        vert = torch.from_numpy(vert).float()

        # out
        res = {'img': roi, 'verts': vert, 'mask': mask, 'root': root, 'calib': calib}

        return res

    def __len__(self):

        return len(self.db_data_anno)

    def visualization(self, res, idx):
        import matplotlib.pyplot as plt
        from mobhand.tools.vis import perspective
        num_sample = 1
        fig = plt.figure(figsize=(8, 2))
        img = inv_based_tranmsform(res['img'].numpy())
        # aligned verts
        ax = plt.subplot(1, 3, 1)
        vert = res['verts'].numpy().copy()
        root = res['root'].numpy().copy()
        vert = vert + root
        proj_vert = perspective(torch.from_numpy(vert.copy()).permute(1, 0).unsqueeze(0), res['calib'].unsqueeze(0))[0].numpy().T
        ax.imshow(img)
        plt.plot(proj_vert[:, 0], proj_vert[:, 1], 'o', color='red', markersize=1)
        ax.set_title('verts')
        ax.axis('off')
        # mask
        ax = plt.subplot(1, 3, 2)
        if res['mask'].ndim == 3:
            mask = res['mask'].numpy()[i] * 255
        else:
            mask = res['mask'].numpy() * 255
        mask_ = np.concatenate([mask[:, :, None]] + [np.zeros_like(mask[:, :, None])] * 2, 2).astype(np.uint8)
        img_mask = cv2.addWeighted(img, 1, mask_, 0.5, 1)
        ax.imshow(img_mask)
        ax.set_title('mask')
        ax.axis('off')

        face = torch.from_numpy( np.load('template/right_faces.npy') ).unsqueeze(0)
        sl = SilhouetteLoss(self.data_size, face)
        l, m = sl(res['verts'].unsqueeze(0), res['mask'].unsqueeze(0))
        ax = plt.subplot(1, 3, 3)
        ax.imshow(m.numpy()[0])
        ax.set_title('render')
        ax.axis('off')

        plt.subplots_adjust(left=0., right=0.95, top=0.95, bottom=0.03, wspace=0.12, hspace=0.1)
        plt.savefig(f'debug/{idx}_{i}.jpg')


if __name__ == '__main__':
    dataset = FreiHAND()
    for i in range(0, len(dataset), len(dataset)//10):
        print(i)
        data = dataset.__getitem__(i)
        dataset.visualization(data, i)
