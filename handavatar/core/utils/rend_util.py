import numpy as np
import imageio
import skimage
import cv2
import torch
from torch.nn import functional as F
import numpy as np

from pytorch3d import renderer
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures


def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb: # [-1,1] --> [0,1]
        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1. ) / 2.

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).cuda())

    return psnr


def load_rgb(path, normalize_rgb = False):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    if normalize_rgb: # [-1,1] --> [0,1]
        img -= 0.5
        img *= 2.
    img = img.transpose(2, 0, 1)
    return img


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def get_camera_for_plot(pose):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:].detach()
        R = quat_to_rot(pose[:,:4].detach())
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        R = pose[:, :3, :3]
    cam_dir = R[:, :3, 2]
    return cam_loc, cam_dir


def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_sphere_intersections(cam_loc, ray_directions, r = 1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3),
                            cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r ** 2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print('BOUNDING SPHERE PROBLEM!')
        exit()

    sphere_intersections = torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).cuda().float() - ray_cam_dot
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections


class Renderer():
    """ Adapted from SNARF """

    @torch.no_grad()
    def __init__(self, image_size=256):
        super().__init__()

        R = torch.from_numpy(np.array([[-1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., -1.]])).float().unsqueeze(0)

        t = torch.from_numpy(np.array([[0., 0.0, 5.]])).float()

        cameras = renderer.FoVOrthographicCameras(R=R, T=t)
        lights = renderer.PointLights(location=[[0.0, 0.0, 3.0]],
                                           ambient_color=((1, 1, 1),), diffuse_color=((0, 0, 0),),
                                           specular_color=((0, 0, 0),))
        raster_settings = renderer.RasterizationSettings(image_size=image_size, faces_per_pixel=100, blur_radius=0)
        rasterizer = renderer.MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        shader = renderer.HardPhongShader(cameras=cameras, lights=lights)
        # shader = renderer.SoftPhongShader(cameras=cameras, lights=lights)
        self.renderer = renderer.MeshRenderer(rasterizer=rasterizer, shader=shader)

    @torch.no_grad()
    def render_mesh(self, verts, faces, colors=None, mode='npat'):
        """
        mode: normal, phong, texture
        """
        mesh = Meshes(verts, faces)

        normals = torch.stack(mesh.verts_normals_list())
        front_light = torch.tensor([0, 0, 1]).float().to(verts.device)
        shades = (normals * front_light.view(1, 1, 3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)
        results = []

        self.renderer.to(verts.device)
        # normal
        if 'n' in mode:
            normals_vis = normals * 0.5 + 0.5
            mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
            image_normal = self.renderer(mesh_normal)
            results.append(image_normal)

        # shading
        if 'p' in mode:
            mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
            image_phong = self.renderer(mesh_shading)
            results.append(image_phong)

        # albedo
        if 'a' in mode:
            assert (colors is not None)
            mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
            image_color = self.renderer(mesh_albido)
            results.append(image_color)

        # albedo*shading
        if 't' in mode:
            assert (colors is not None)
            mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors * shades))
            image_color = self.renderer(mesh_teture)
            results.append(image_color)

        return torch.cat(results, axis=1)