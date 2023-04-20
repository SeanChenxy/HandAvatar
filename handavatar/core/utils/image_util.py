import os
import shutil

from termcolor import colored
from PIL import Image
import numpy as np
from handavatar.core.utils.geo_util import save_obj


def load_image(path, to_rgb=True):
    img = Image.open(path)
    return img.convert('RGB') if to_rgb else img


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def to_8b_image(image):
    return (255.* np.clip(image, 0., 1.)).astype(np.uint8)


def to_3ch_image(image):
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        assert image.shape[2] == 1
        return np.concatenate([image, image, image], axis=-1)
    else:
        print(f"to_3ch_image: Unsupported Shapes: {len(image.shape)}")
        return image


def to_8b3ch_image(image):
    return to_3ch_image(to_8b_image(image))


def tile_images(images, imgs_per_row=4):
    rows = []
    row = []
    imgs_per_row = min(len(images), imgs_per_row)
    for i in range(len(images)):
        row.append(images[i])
        if len(row) == imgs_per_row:
            rows.append(np.concatenate(row, axis=1))
            row = []
    if len(rows) > 2 and len(rows[-1]) != len(rows[-2]):
        rows.pop()
    imgout = np.concatenate(rows, axis=0)
    return imgout

     
class ImageWriter():
    def __init__(self, output_dir, exp_name, clear=True):
        self.image_dir = os.path.join(output_dir, exp_name)

        print("The rendering is saved in " + \
              colored(self.image_dir, 'cyan'))
        
        # remove image dir if it exists
        if os.path.exists(self.image_dir) and clear:
            shutil.rmtree(self.image_dir)
        
        os.makedirs(self.image_dir, exist_ok=True)
        self.frame_idx = -1

    def append(self, image, img_name=None):
        self.frame_idx += 1
        if img_name is None:
            img_name = f"{self.frame_idx:06d}"
        save_image(image, f'{self.image_dir}/{img_name}.png')
        return self.frame_idx, img_name
    
    def append_obj(self, verts, faces, obj_name=None):
        if obj_name is None:
            obj_name = f"{self.image_dir}/{self.frame_idx:06d}.obj"
        save_obj(verts, faces, obj_name)
    
    def append_ply(self, mesh, ply_name=None, postfix=None):
        if ply_name is None:
            ply_name = f'{self.image_dir}/{self.frame_idx:06d}'
        if postfix is not None:
            ply_name += '_' + postfix
        ply_name += '.obj'
        mesh.export(ply_name)

    def finalize(self):
        pass
    
    def skip(self):
        self.frame_idx += 1
