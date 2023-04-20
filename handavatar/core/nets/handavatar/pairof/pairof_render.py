import os
import copy

import smplx
import torch
import trimesh
import numpy as np
import torch.nn.functional as F

from skimage import measure
from pytorch3d.structures import Meshes
# from pytorch3d.ops import sample_points_from_meshes

from pairof.utils.sample_points_from_meshes import sample_points_from_meshes
from pairof.pairof_model.modules import ResnetPointnet, ImplicitNet

JOINT_NAMES = [ 'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky', 'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 'left_eye_brow5', 'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1', 'nose1', 'nose2', 'nose3', 'nose4', 'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2', 'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6', 'left_eye4', 'left_eye3', 'left_eye2', 'left_eye1', 'left_eye6', 'left_eye5', 'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'mouth_top', 'left_mouth_3', 'left_mouth_2', 'left_mouth_1', 'left_mouth_5',  'mouth_bottom', 'right_mouth_4', 'right_mouth_5', 'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3', 'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1']
class Partitioner(torch.nn.Module):
    MERGE_BODY_PARTS = {
        'smpl': sorted([
            15,  # face into neck
            10,  # left toes into left foot
            11,  # right toes into right foot
            3,  # mid stomach into lower stomach
            13, 14,  # left and right shoulder blade
            22, 23,  # left and right hand into wrists
        ], reverse=True),

        'smplx': sorted([
            15,  # face into neck
            10,  # left toes into left foot
            11,  # right toes into right foot
            3,  # mid stomach into lower stomach
            13, 14,  # left and right shoulder blade
            9,  # upper body with upper stomach
        ], reverse=True),
        'mano': []
    }
    SELFPEN_DISABLE_PARTS = {  # disable close connections for self-penetration
        'smpl': [
            (1, 2),  # L_Hip - R_Hip

            (1, 3),  # L_Hip - Spine1
            (2, 3),  # R_Hip - Spine1

            (3, 9),  # Spine1 - Spine3

            (9, 15),  # Spine3 - Head 
            (9, 16),  # Spine3 - L_Shoulder 
            (9, 17),  # Spine3 - R_Shoulder 

            (12, 6),  # Neck - Spline2
            (12, 16),  # Neck - L_Shoulder
            (12, 17),  # Neck - R_Shoulder

            (13, 6),  # L_Collar - Spine2
            (13, 12),  # L_Collar - Neck
            (14, 6),  # R_Collar - Spine2
            (14, 12),  # R_Collar - neck

            (0, 6),  # pelvis and spine2
            (0, 9),  # pelvis and spine3
            (0, 16),  # pelvis and L_Shoulder 
            (0, 17),  # pelvis and R_Shoulder 

            (6, 16),  # Spine2 - L_Shoulder
            (6, 17),  # Spine2 - R_Shoulder
        ],
        'smplx': [
            (1, 2),  # L_Hip - R_Hip

            (1, 3),  # L_Hip - Spine1
            (2, 3),  # R_Hip - Spine1

            (3, 9),  # Spine1 - Spine3

            (9, 15),  # Spine3 - Head
            (9, 16),  # Spine3 - L_Shoulder
            (9, 17),  # Spine3 - R_Shoulder

            (12, 6),  # Neck - Spline2
            (12, 16),  # Neck - L_Shoulder
            (12, 17),  # Neck - R_Shoulder

            (13, 6),  # L_Collar - Spine2
            (13, 12),  # L_Collar - Neck
            (14, 6),  # R_Collar - Spine2
            (14, 12),  # R_Collar - neck

            (0, 6),  # pelvis and spine2
            (0, 9),  # pelvis and spine3
            (0, 16),  # pelvis and L_Shoulder
            (0, 17),  # pelvis and R_Shoulder

            (6, 16),  # Spine2 - L_Shoulder
            (6, 17),  # Spine2 - R_Shoulder
        ],
        'mano': []
    }

    def __init__(self, parametric_body: smplx.SMPL, smpl_cfg: dict):
        super().__init__()

        self.model_type = self.get_model_type(parametric_body)
        self.merge_body_parts = self.MERGE_BODY_PARTS[self.model_type]
        self.selfpen_disable_parts = self.SELFPEN_DISABLE_PARTS[self.model_type]

        self.extended_dist = smpl_cfg['extended_dist']
        (tight_face_tensor, extended_face_tensor), tight_vert_selector = self.partition_body(parametric_body)

        self.num_parts = len(tight_vert_selector)
        joint_mapper = torch.ones(self.num_parts+len(self.merge_body_parts), dtype=torch.bool)
        joint_mapper[self.merge_body_parts] = False

        # register variables
        self.faces = parametric_body.faces.copy()
        self.register_buffer('parents', parametric_body.parents.detach().clone().long())
        self.register_buffer('joint_mapper', joint_mapper)
        self.register_buffer('tight_face_tensor', tight_face_tensor)
        self.register_buffer('extended_face_tensor', extended_face_tensor)
        self.register_buffer('tight_vert_selector', tight_vert_selector)

        # variables for resolving self-intersections
        selfpen_disable_mat = self.get_selfpen_disable_mat(joint_mapper, (24, 16)[self.model_type=='mano'])
        self.register_buffer('selfpen_disable_mat', selfpen_disable_mat, persistent=False)

        # global sample
        sample_points, (sample_face_idxs, w0, w1, w2) = self.global_sample_mesh_points(parametric_body.v_template, parametric_body.faces_tensor, smpl_cfg['global_sample_num'])
        self.register_buffer('sample_face_tensor', parametric_body.faces_tensor[sample_face_idxs[0]].long())
        self.register_buffer('w0', w0)
        self.register_buffer('w1', w1)
        self.register_buffer('w2', w2)
        joints = torch.mm( parametric_body.J_regressor, parametric_body.v_template )
        self.register_buffer('sample_points', sample_points[0] - joints[4:5])

    @torch.no_grad()
    def get_selfpen_disable_mat(self, joint_mapper, n_parts=24):
        # smpl_body = np.load(bm_path)
        # kintree_table = smpl_body['kintree_table'][0].astype(np.int32)[:n_parts]
        kintree_table = self.parents.cpu().numpy()[:n_parts]
        mat = np.ones((n_parts, n_parts), dtype=np.float32)
        for b_part in range(n_parts):
            # disable offsprings
            children_ind = [i for i in range(n_parts) if kintree_table[i] == b_part]
            mat[b_part, children_ind] = 0

            # disable parent nodes
            for ch in children_ind:
                parent_id = kintree_table[ch]
                if parent_id != -1:
                    mat[ch, parent_id] = 0

        for elem in self.selfpen_disable_parts:
            mat[elem[0], elem[1]] = 0
            mat[elem[1], elem[0]] = 0
        
        mat = torch.from_numpy(mat)
        jm_inds = torch.where(joint_mapper)
        mat = mat[jm_inds].T[jm_inds].T

        return mat.bool()

    def check_bbox_penetrations(self, bb_min, bb_max, abs_transforms, EPS=1e-4):
        """
        Args:
            bb_min (torch.tensor): K,3
            bb_max (torch.tensor): K,3
            abs_transforms (torch.tensor): K,4,4
            disable_grad_mat (torch.tensor): K,K
        Returns:
        """
        K = bb_min.shape[0]
        m = bb_min.view(K, 3, 1)
        x = bb_max.view(K, 3, 1)
        corners = torch.cat([
            torch.cat((m[:, 0], m[:, 1], m[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((m[:, 0], m[:, 1], x[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((m[:, 0], x[:, 1], m[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((m[:, 0], x[:, 1], x[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((x[:, 0], m[:, 1], m[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((x[:, 0], m[:, 1], x[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((x[:, 0], x[:, 1], m[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
            torch.cat((x[:, 0], x[:, 1], x[:, 2]), dim=-1).unsqueeze(-2),  # K,1,3
        ], dim=-2)  # K,8,3

        hcorners = torch.ones((K * 8, 4), dtype=bb_min.dtype, device=bb_min.device)
        hcorners[:, :3] = corners.view(-1, 3)
        hposed_corners = torch.bmm(
            abs_transforms.view(K, 1, 4, 4).repeat(1, 8, 1, 1).view(K * 8, 4, 4),
            hcorners.view(K * 8, 4, 1)
        ).view(K * 8, 4)

        # project posed corners to each other's space
        back_trans = torch.inverse(abs_transforms)

        proj_corners = torch.bmm(
            back_trans.view(K, 1, 4, 4).repeat(1, K * 8, 1, 1).view(K * K * 8, 4, 4),  # K*K*8,4,4
            hposed_corners.view(1, K * 8, 4).repeat(K, 1, 1).view(K * K * 8, 4, 1)  # K*K*8,4,1
        ).view(K, K, 8, 4)[..., :3]  # K, K, 8, 3

        # check which corner is inside of which box
        bb_min = bb_min.view(K, 1, 1, 3).repeat(1, K, 8, 1)  # (K, K, 8, 3)
        bb_max = bb_max.view(K, 1, 1, 3).repeat(1, K, 8, 1)  # (K, K, 8, 3)

        inside = ((proj_corners > bb_min + EPS) & (proj_corners < bb_max - EPS)).all(dim=-1)  # K, K, 8
        inside = inside.any(dim=-1)  # at least one corner needs to be inside
        # print('Prev collisions', inside.sum())
        inside = inside & self.selfpen_disable_mat

        # remove duplicates
        inside = inside | inside.t()  # symmetric table
        inside = torch.tril(inside)  # lower triangular part
        # print('After collisions', inside.sum())

        # assert torch.allclose(inside, inside.t())  # ensure this check is symmetric
        inds = torch.stack(torch.where(inside)).sort(dim=0)[0].t()
        # print(inds.cpu().numpy())
        return hposed_corners.view(K, 8, 4)[..., :3], inds

    @staticmethod
    def sample_collision_candidates(bb_min, bb_max, abs_transforms, collided_inds, n_points_uniform=300):
        """
        Args:
            bb_min (torch.tensor): K,3
            bb_max (torch.tensor): K,3
            abs_transforms (torch.tensor): K,4,4
            collided_inds (torch.tensor): N,2
            n_points_uniform (int):
        Returns:
        """
        def _check_inside_bbox(points, bwd_trans, box_min, box_max):  # (N,P,3), (N,4,4), (N,3)
            can_points = (bwd_trans.reshape(-1, 1, 4, 4).expand(-1, points.size(1), -1, -1) @ F.pad(points, [0, 1], "constant", 1.0).unsqueeze(-1))[:, :, :3, 0]
            p_in = ((can_points > box_min.unsqueeze(1)) & (can_points < box_max.unsqueeze(1))).all(dim=-1)
            return p_in  # (N, P)

        bb_min, bb_max = bb_min.reshape(-1, 3), bb_max.reshape(-1, 3)
        collided_inds = collided_inds.reshape(-1)  # (N*2,)

        c_bb_min = bb_min[collided_inds]  # (N*2,3)
        c_bb_max = bb_max[collided_inds]  # (N*2,3)
        bbox_size = (c_bb_max - c_bb_min)  # (N*2,3)
        uniform_points = torch.rand((bbox_size.size(0), n_points_uniform, 3), device=bb_min.device)  # (N*2,P,3)
        uniform_points = uniform_points * bbox_size.unsqueeze(1) + c_bb_min.unsqueeze(1)  # (N*2,P,3)

        # project points to the posed space
        fwd_trans = abs_transforms[collided_inds]  # (N*2, 4, 4)
        uniform_points = torch.matmul(
            fwd_trans.reshape(-1, 2, 1, 4, 4).expand(-1, -1, n_points_uniform, -1, -1),  # (N,2,P,4,4)
            F.pad(uniform_points, [0, 1], "constant", 1.0).reshape(-1, 2, n_points_uniform, 4, 1)  # (N,2, P, 4, 1)
        )[..., :3, 0]  # (N, 2, P, 3)

        # keep only samples that are inside both colliding boxes
        bwd_trans = torch.inverse(abs_transforms)[collided_inds].view(-1, 2, 4, 4)  # (N, 2, 4, 4)
        c_bb_min, c_bb_max = c_bb_min.view(-1, 2, 3), c_bb_max.view(-1, 2, 3)  # (N, 2, 3)

        in1 = _check_inside_bbox(uniform_points[:, 0, ...], bwd_trans[:, 1, ...], c_bb_min[:, 1, ...], c_bb_max[:, 1, ...])
        in2 = _check_inside_bbox(uniform_points[:, 1, ...], bwd_trans[:, 0, ...], c_bb_min[:, 0, ...], c_bb_max[:, 0, ...])

        collision_candidates1 = uniform_points[:, 0, ...][in1]
        collision_candidates2 = uniform_points[:, 1, ...][in2]
        collision_candidates = torch.cat((collision_candidates1, collision_candidates2), dim=0)
        return collision_candidates

    @staticmethod
    def get_model_type(parametric_body):
        if type(parametric_body) == smplx.MANO:
            model_type = 'mano'
        elif type(parametric_body) == smplx.FLAME:
            model_type = 'flame'
        elif type(parametric_body) == smplx.SMPLX:
            model_type = 'smplx'
        elif type(parametric_body) == smplx.SMPLH:
            model_type = 'smplh'
        elif type(parametric_body) == smplx.SMPL:
            model_type = 'smpl'
        else:
            raise NotImplementedError('The body type is not supported.')
        return model_type

    def get_bbox_bounds(self, vertices, bone_trans):
        """
        Args:
            vertices (torch.tensor): (B,V,3)
            bone_trans (torch.tensor): (B,K,4,4)
        Returns:
            tuple:
                bbox_min (torch.tensor): BxKx1x3
                bbox_max (torch.tensor): BxKx1x3
        """
        B = vertices.shape[0]
        K, max_select_dim = self.tight_vert_selector.shape[:2]

        part_vertices = torch.index_select(vertices, 1, self.tight_vert_selector.view(-1))  # (B, K*max_select_dim, 3)
        part_vertices = part_vertices.view(B, K, max_select_dim, 3)

        if bone_trans is not None:
            bone_trans = bone_trans.unsqueeze(2).expand(-1, -1, max_select_dim, -1, -1)
            local_part_vertices = (bone_trans @ F.pad(part_vertices, [0, 1], "constant", 1.0).unsqueeze(-1))[..., :3, 0]
        else:
            local_part_vertices = part_vertices
        bbox_min = local_part_vertices.min(dim=-2, keepdim=True).values  # (B, K, 1, 3)
        bbox_max = local_part_vertices.max(dim=-2, keepdim=True).values  # (B, K, 1, 3)

        return bbox_min, bbox_max

    @torch.no_grad()
    def partition_body(self, parametric_body):
        # prepare data
        faces = self.tensor_to_numpy(parametric_body.faces).copy()
        v_template = self.tensor_to_numpy(parametric_body.v_template).copy()
        lbs_weights = self.tensor_to_numpy(parametric_body.lbs_weights).copy()
        parents = self.tensor_to_numpy(parametric_body.parents.long()).copy()

        max_part = lbs_weights.argmax(axis=1)
        can_mesh = trimesh.Trimesh(v_template, faces, process=False)
        kintree_table = parents

        # merge parts
        if self.model_type == 'smplx':
            # merge jaw into face and hand articulation
            max_part[(max_part == 22)] = kintree_table[22]  # jaw
            max_part[(max_part >= 25) & (max_part < 40)] = kintree_table[25 + 3]  # merge left hand
            max_part[(max_part >= 40) & (max_part < 55)] = kintree_table[40 + 3]  # merge right hand
            max_part[max_part >= 22] = 100  # invalid labels for eyes
            n_parts = 22
        elif self.model_type == 'smpl':
            n_parts = 24
        elif self.model_type == 'mano':
            n_parts = 16
        else:
            n_parts = kintree_table.shape[0]

        kintree_table = kintree_table[:n_parts]
        valid_labels = list(range(n_parts))
        
        # merge small body parts
        for ind in self.merge_body_parts:
            # relabel vertices
            max_part[(max_part == ind)] = kintree_table[ind]
            valid_labels[ind] = kintree_table[ind]

            # update kintable; all children take a grandparent's id
            kintree_table[kintree_table == ind] = kintree_table[ind]

        # # convert vertex labels
        # face labels
        edges = can_mesh.edges_unique
        faces_unique_edges = can_mesh.faces_unique_edges  # F,3
        lab_list = []
        for f_ind in range(can_mesh.faces.shape[0]):
            vert_id = edges[faces_unique_edges[f_ind]].reshape(-1)  # (6,)
            vert_id = np.unique(vert_id)  # (3,)
            vert_sk_lab = max_part[vert_id]  # (3,)
            lab_list.append(np.bincount(vert_sk_lab).argmax())

        face_labels = np.array(lab_list)

        tight_face_list, extended_face_list = [], []
        tight_face_mask_list, extended_face_mask_list = [], []
        tight_vertex_mask_list, extended_vertex_mask_list = [], []
        for b_part in range(n_parts):
            if b_part in self.merge_body_parts:
                continue

            b_label = valid_labels[b_part]
            tight_inds, extended_inds = face_labels == b_label, face_labels == b_label
            parent_ind = kintree_table[b_part]
            ch_inds = [i for i in range(n_parts) if kintree_table[i] == b_part]
            ch_label = [valid_labels[ch] for ch in ch_inds]

            if parent_ind != -1:
                parent_label = valid_labels[parent_ind]
                extended_inds = extended_inds | (face_labels == parent_label)
            for ch_lab in ch_label:
                extended_inds = extended_inds | (face_labels == ch_lab)

            extended_face = faces[extended_inds]
            tight_face = faces[tight_inds]
            tight_verts = v_template[np.unique(tight_face)]
            extended_verts = v_template[extended_face.reshape(-1)]
            minidist = (extended_verts[:, None] - tight_verts[None])
            minidist = np.linalg.norm(minidist, axis=-1)
            minidist = minidist.min(axis=-1).reshape(extended_face.shape[0], extended_face.shape[1])
            sel_extended = (minidist < self.extended_dist).all(axis=1)
            # print(b_part, sel_extended.shape, sel_extended.sum())
            extended_inds[extended_inds] = sel_extended

            tight_face_list.append(can_mesh.faces[tight_inds])
            tight_face_mask_list.append(tight_inds)
            extended_face_list.append(can_mesh.faces[extended_inds])
            extended_face_mask_list.append(extended_inds)

            tight_vertex_inds, extended_vertex_inds = max_part == b_label, max_part == b_label
            if parent_ind != -1:
                extended_vertex_inds = extended_vertex_inds | (max_part == parent_label)
            for ch_lab in ch_label:
                extended_vertex_inds = extended_vertex_inds | (max_part == ch_lab)

            tight_vertex_mask_list.append(tight_vertex_inds)
            extended_vertex_mask_list.append(extended_vertex_inds)

        tight_face_tensor, extended_face_tensor = [], []
        tight_max_size = max([x.shape[0] for x in tight_face_list])
        extended_max_size = max([x.shape[0] for x in extended_face_list])
        for x in tight_face_list:
            padded_face_tensor = np.full((tight_max_size, 3), fill_value=-1, dtype=np.int32)
            padded_face_tensor[:x.shape[0], :] = x
            tight_face_tensor.append(torch.from_numpy(padded_face_tensor).unsqueeze(0))

        for x in extended_face_list:
            padded_face_tensor = np.full((extended_max_size, 3), fill_value=-1, dtype=np.int32)
            padded_face_tensor[:x.shape[0], :] = x
            extended_face_tensor.append(torch.from_numpy(padded_face_tensor).unsqueeze(0))

        tight_face_tensor = torch.cat(tight_face_tensor, dim=0)
        extended_face_tensor = torch.cat(extended_face_tensor, dim=0)
        tight_vert_selector = [
            torch.where(torch.from_numpy(max_part == valid_labels[i]))[0]
            for i in range(n_parts) if i not in self.merge_body_parts
        ]
        # tight_vert_selector = [torch.from_numpy(vert_ids) for vert_ids in max_part]
        max_elem = max([v_sec.shape[0] for v_sec in tight_vert_selector])
        for b_part in range(len(tight_vert_selector)):
            vert_sec = torch.full((max_elem,), dtype=tight_vert_selector[b_part].dtype,
                                fill_value=tight_vert_selector[b_part][0].item())
            vert_sec[:tight_vert_selector[b_part].shape[0]] = tight_vert_selector[b_part]
            tight_vert_selector[b_part] = vert_sec

        tight_vert_selector = torch.stack(tight_vert_selector)
        return (tight_face_tensor, extended_face_tensor), tight_vert_selector

    @staticmethod
    def _sample_mesh_points(posed_vert, face_tensor, n_samples):
        B, V, _ = posed_vert.shape
        n_parts = face_tensor.shape[0]
        meshes = Meshes(  # (B*n_parts,V,3)
            verts=posed_vert.unsqueeze(1).expand(-1, n_parts, -1, -1).contiguous().view(B * n_parts, -1, 3),
            faces=face_tensor.unsqueeze(0).expand(B, -1, -1, -1).contiguous().view(B * n_parts, -1, 3))
        samp_points, samp_normals = sample_points_from_meshes(meshes, n_samples, return_normals=True)
        return samp_points.view(B, n_parts, n_samples, 3), samp_normals.view(B, n_parts, n_samples, 3)
    
    def global_sample_mesh_points(self, verts, faces, n_samples):
        meshes = Meshes(
            verts=verts.unsqueeze(0),
            faces=faces.unsqueeze(0))
        samp_points, samp_info = sample_points_from_meshes(meshes, n_samples, return_sample_info=True)
        # pts = samp_points[0].detach().cpu().numpy()
        # rgb = np.ones_like(pts)
        # to_save = np.concatenate([pts, rgb * 255], axis=-1)
        # np.savetxt(f'/mnt/user/chenxingyu/jupyter/gl3.ply',
        #           to_save,
        #           fmt='%.6f %.6f %.6f %d %d %d',
        #           comments='',
        #           header=(
        #               'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
        #               pts.shape[0])
        #           )
        return samp_points, samp_info

    def sample_mesh_points(self, posed_vert, n_samples):
        n_tight_samples = n_samples // 2
        n_ext_samples = n_samples - n_tight_samples #[ns - nt for ns, nt in zip(n_samples, n_tight_samples)] #n_samples - n_tight_samples
        samp_points, samp_normals = self._sample_mesh_points(
            posed_vert,
            self.tight_face_tensor,
            n_tight_samples)
        if n_ext_samples > 0:
            uniform_points, uniform_normals = self._sample_mesh_points(
                posed_vert,
                self.extended_face_tensor,
                n_ext_samples)
            samp_points = torch.cat((samp_points, uniform_points), dim=-2)
            samp_normals = torch.cat((samp_normals, uniform_normals), dim=-2)

        return samp_points, samp_normals

    def compute_abs_transformations(self, full_pose, posed_joints):
        # full_pose: B, K*3 or B,K,3
        # posed_joints: B, K*3 or B,K,3
        B = full_pose.shape[0]
        full_pose = full_pose.reshape(B, -1, 3)
        posed_joints = posed_joints.reshape(B, -1, 3)
        K = full_pose.shape[1]
        mK = self.joint_mapper.shape[0]
        # posed_joints = posed_joints[:, :K]  # remove extra joints (landmarks for smplx)
        
        # torchgeometry.angle_axis_to_rotation_matrix(full_pose.view(-1, 3))[:, :3, :3]
        rot_mats = smplx.lbs.batch_rodrigues(full_pose.view(-1, 3)).view(B, K, 3, 3)  # B*K,3,3

        # fwd lbs to estimate absolute transformation matrices
        parents = self.parents.long()
        transform_chain = [rot_mats[:, 0]]
        for i in range(1, parents.shape[0]):
            if i == mK:
                break
            transform_chain.append(transform_chain[parents[i]] @ rot_mats[:, i])

        transforms = torch.stack(transform_chain, dim=1)
        abs_trans = torch.cat([
            F.pad(transforms.reshape(-1, 3, 3), [0, 0, 0, 1]),
            F.pad(posed_joints[:, :mK].reshape(-1, 3, 1), [0, 0, 0, 1], value=1)
        ], dim=2).reshape(B, mK, 4, 4)

        # remap joints
        abs_trans = abs_trans.transpose(0, 1)[self.joint_mapper].transpose(0, 1)
        return abs_trans

    @staticmethod
    def tensor_to_numpy(tensor_vec):
        if torch.is_tensor(tensor_vec):
            return tensor_vec.detach().cpu().numpy()
        return tensor_vec



class COAPBodyModel(torch.nn.Module):

    def __init__(self, parametric_body: smplx.SMPL, smpl_cfg: dict) -> None:
        super().__init__()

        # hyperparameters
        self.n_samples = smpl_cfg['n_samples']
        self.bbox_padding = smpl_cfg['bbox_padding']  # bounding box size

        # create differentiable modules
        self.partitioner = Partitioner(parametric_body, smpl_cfg)
        self.part_encoder = ResnetPointnet(out_dim=smpl_cfg['part_en_dim'], hidden_dim=smpl_cfg['part_en_dim'], dim=(3, 6)[smpl_cfg['en_part_normal']], final_pool=True)
        self.local_encoder = ResnetPointnet(out_dim=smpl_cfg['local_en_dim'], hidden_dim=smpl_cfg['local_en_dim'], dim=smpl_cfg['part_en_dim']+3, final_pool=smpl_cfg['fusion_pool'])

        self.decoder = ImplicitNet(
            d_in=smpl_cfg['local_en_dim']+(0,2)[smpl_cfg['inbox_feat']],
            d_out=(1, 2)[smpl_cfg['fusion_pool'] ],
            dims=[smpl_cfg['decoder_layer'][1],]*smpl_cfg['decoder_layer'][0],
            skip_in=[smpl_cfg['decoder_layer'][0]//2],
        )
        if smpl_cfg['pre_train'] is not None and os.path.exists(smpl_cfg['pre_train']):
            state_dict = torch.load(smpl_cfg['pre_train'], map_location='cpu')['state_dict']
            self.my_load_state_dict(self, state_dict)
            # self.load_state_dict(state_dict)
            print('Load pre-trained model: {}'.format(smpl_cfg['pre_train']))

        self.level_set = smpl_cfg['level_set']
        self.out_act = torch.sigmoid
        self.impl_code = None
        self.model_type = self.partitioner.model_type
        self.sigma = smpl_cfg['sigma']
        self.softmax_out = smpl_cfg['softmax_out']
        self.inbox_feat = smpl_cfg['inbox_feat']
        self.en_part_normal = smpl_cfg['en_part_normal']
        self.neighbor = smpl_cfg['neighbor']
        self.fusion_pool = smpl_cfg['fusion_pool'] 
        connect = torch.tensor([
            [0, 1], [1, 2], [2, 3], 
            [0, 4], [4, 5], [5, 6],
            [0, 7], [7, 8], [8, 9],
            [0, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15]]).view(-1).long()
        self.register_buffer('connect', connect)
        if smpl_cfg['soft_beta'] < 1:
            self.soft_act = lambda x: 1 / (1+torch.exp(-smpl_cfg['soft_beta']*x))
        else:
            self.soft_act = None
    
    @staticmethod
    def my_load_state_dict(curr_model, prev_model, start=0, end=None):
        own_state = curr_model.state_dict()
        prev_state = prev_model

        for name, param in list(prev_state.items())[start:end]:
            
            if name.startswith('module'):
                name = name[7:]
            if 'partitioner' in name:
                print('ignore:', name)
                continue
            if name not in own_state:
                raise KeyError('Unexpected key "{}" in state_dict'.format(name))
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
                print('copy:', name)
            except:
                pad = [2,] * param.ndim
                param = torch.tile(param, pad)
                if param.ndim==1:
                    own_state[name].copy_(param[:own_state[name].shape[0]])
                else:
                    own_state[name].copy_(param[:own_state[name].shape[0], :own_state[name].shape[1]])
                print('Note: the parameter {} is inconsistent! but we pad it for copy'.format(name))
                continue

    def get_bbox_bounds(self, vertices, bone_trans):
        bbox_min, bbox_max = self.partitioner.get_bbox_bounds(vertices, bone_trans)  # (B, K, 1, 3)
        return bbox_min, bbox_max
    
    def get_tight_face_tensor(self):
        return self.partitioner.tight_face_tensor.contiguous()

    def compute_bone_trans(self, full_pose, joints):
        abs_transforms = self.partitioner.compute_abs_transformations(full_pose, joints)
        bone_trans = torch.inverse(abs_transforms)
        return bone_trans

    def encode_body(self, smpl_output):
        full_pose = smpl_output.full_pose
        joints = smpl_output.joints
        vertices = smpl_output.vertices

        # estimate absolute transformation matrices
        bone_trans = self.compute_bone_trans(full_pose, joints)
        bbox_min, bbox_max = self.get_bbox_bounds(vertices, bone_trans)  # (B, K, 1, 3)

        # estimate absolute transformation matricess
        sampled_points, sampled_normals = self.partitioner.sample_mesh_points(vertices, self.n_samples)
        B, K, n_points, _ = sampled_points.shape

        # canonicalize sampled points [B,K,T,3]
        local_sampled_points = (bone_trans.reshape(B, K, 1, 4, 4).expand(-1, -1, n_points, -1, -1) @ F.pad(sampled_points, [0, 1], "constant", 1.0).unsqueeze(-1))[..., :3, 0]  # B,K,T,3
        part_input = [local_sampled_points.reshape(B*K, n_points, 3),]
        if self.en_part_normal:
            local_sampled_normals = (bone_trans.reshape(B, K, 1, 4, 4).expand(-1, -1, n_points, -1, -1) @ F.pad(sampled_normals, [0, 1], "constant", 0.0).unsqueeze(-1))[..., :3, 0]
            part_input.append(local_sampled_normals.reshape(B*K, n_points, 3))
        # encode local point clouds
        part_input = torch.cat(part_input, -1)
        latent_code = self.part_encoder(part_input)
        latent_code = latent_code.view(B, K, -1) 
        latent_code = torch.index_select(latent_code, 1, self.connect).view(B, K-1, 2, -1)

        # global points
        face_verts = vertices[:, self.partitioner.sample_face_tensor]
        v0, v1, v2 = face_verts[:, :, 0], face_verts[:, :, 1], face_verts[:, :, 2]
        global_points = self.partitioner.w0[:, :, None] * v0 + self.partitioner.w1[:, :, None] * v1 + self.partitioner.w2[:, :, None] * v2

        # store data
        self.impl_code = dict(
            full_pose=full_pose,
            joints=joints,
            bone_trans=bone_trans,
            latent_code=latent_code,
            global_points=global_points,
            bbox_min=bbox_min, bbox_max=bbox_max,
            bbox_size=torch.abs(bbox_max - bbox_min)*self.bbox_padding,
            bbox_center=(bbox_min + bbox_max)*0.5  # (B, K, 1, 3),
        )

    def _attach_impl_code(self, smpl_output=None):
        if smpl_output is None:
            return

        # recalculate encodings if input has changed
        if self.impl_code is None:
            assert smpl_output is not None
            self.encode_body(smpl_output)
        elif smpl_output.full_pose.shape != self.impl_code['full_pose'].shape or\
            not torch.allclose(smpl_output.full_pose, self.impl_code['full_pose']) or\
            not torch.allclose(smpl_output.joints, self.impl_code['joints']):
            self.encode_body(smpl_output)

    def detach_cache(self):
        self.impl_code = None

    @staticmethod
    def masked_softmax(vec, mask, dim=1, sigma=0.5, epsilon=1e-5):
        s_vec = vec.clone()
        # s_vec[s_vec < 0.5] -= 0.5
        exps = torch.exp(s_vec/sigma)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        weight = (masked_exps/masked_sums)
        out = (weight * vec).sum(dim)
        return out

    def query(self, points, smpl_output=None, ret_intermediate=False):
        """
        Args:
            points (torch.tensor): Query points of shape [B,T,3]
        """
        # attach variables
        # import time
        # t1 = time.time()
        self._attach_impl_code(smpl_output)

        latent_code = self.impl_code['latent_code']
        global_points = self.impl_code['global_points']
        bone_trans = self.impl_code['bone_trans']
        bbox_size = self.impl_code['bbox_size']
        bbox_center = self.impl_code['bbox_center']

        B, C = latent_code.shape[:2]
        K = bone_trans.shape[1]
        T = points.shape[1]
        # t2 = time.time()

        # global relation
        gdists, gindices = (points[:, :, None, :] - global_points[:, None, :, :]).norm(dim=-1).topk(self.neighbor, dim=-1, largest=False)
        gdists = torch.clamp(gdists, min=0.0001)
        # t3 = time.time()

        # canonicalize query points
        local_queries = (bone_trans.reshape(B, K, 1, 4, 4).expand(-1, -1, T, -1, -1) @ F.pad(points, [0, 1], "constant", 1.0).reshape(B, 1, T, 4, 1).expand(-1, K, -1, -1, -1))[..., :3, 0]
        inside_bbox = ((local_queries - bbox_center).abs() < (bbox_size * 0.5)).all(dim=-1).int() # (B, K, T)
        inside_connect = torch.index_select(inside_bbox, 1, self.connect).view(B, C, 2, T).permute(0,1,3,2)
        valid = inside_connect.any(dim=-1).view(B*C*T).bool()
        # t4 = time.time()
        if valid.sum() == 0:
            occupancy = torch.zeros([B, T]).to(points.device)
            part_occupancy = torch.zeros([B, K, T]).to(points.device)
            part_pred = {'part_occupancy': part_occupancy, 'all_out': ~inside_bbox.any(1), 'gdists': gdists, 'gindices': gindices}
            if self.soft_act is not None:
                soft_part_occupancy = torch.zeros([B, K, T]).to(points.device)
                part_pred.update({'soft_part_occupancy': soft_part_occupancy})
                if not self.training:
                    soft_occupancy = torch.zeros([B, T]).to(points.device)
                    part_pred.update({'soft_occupancy': soft_occupancy})
            if ret_intermediate:
                return occupancy, part_pred
            return occupancy

        query_feat = []
        if self.inbox_feat:
            query_feat += [inside_connect.reshape(B*C*T, -1)[valid]]

        latent_code = latent_code[:,:,None].expand(-1, -1, T, -1, -1)
        local_queries = torch.index_select(local_queries, 1, self.connect).view(B, C, 2, T, -1).permute(0, 1, 3, 2, 4)
        local_feat = torch.cat([latent_code, local_queries], -1).reshape(B*C*T, 2, -1)[valid]
        local_feat = self.local_encoder(local_feat)
        query_feat += [local_feat, ]

        # blend features and local queries
        z_code = torch.cat(query_feat, dim=-1)
        # print('local_feat: ', local_feat.shape, z_code.shape)
        # t5 = time.time()

        # query occupancy
        connect_occupancy = torch.zeros([B*C*T, 2]).to(points.device)
        decode_out = self.decoder(z_code)
        if not self.fusion_pool:
            decode_out = decode_out[..., 0]
        connect_occupancy[valid] = self.out_act(-decode_out)
        # t6 = time.time()
        connect_occupancy = connect_occupancy.reshape(B, C, T, 2)
        connect_occupancy = connect_occupancy.permute(1, 3, 0, 2).reshape(-1, B, T)
        part_occupancy = torch.zeros([B, C, T, K]).to(points.device)
        dimC = torch.arange(0, C)[:,None].tile(1,2).view(-1).long().to(points.device)
        part_occupancy[:, dimC, :, self.connect] = connect_occupancy
        if self.softmax_out:
            part_occupancy = self.masked_softmax(part_occupancy, part_occupancy>0.0, dim=1, sigma=self.sigma)
        else:
            part_occupancy = part_occupancy.max(dim=1).values
        part_occupancy = part_occupancy.permute(0,2,1) * inside_bbox
        if self.softmax_out:
            occupancy = self.masked_softmax(part_occupancy, part_occupancy>0.0, dim=1, sigma=self.sigma)
        else:
            occupancy = part_occupancy.max(dim=1).values  # (B, T)
        
        part_pred = {'part_occupancy': part_occupancy, 'all_out': ~inside_bbox.any(1), 'gdists': gdists, 'gindices': gindices}
        # t7 = time.time()
        if self.soft_act is not None:
            soft_connect_occupancy = torch.zeros([B*C*T, 2]).to(points.device)
            soft_connect_occupancy[valid] = self.soft_act(-decode_out)
            soft_connect_occupancy = soft_connect_occupancy.reshape(B, C, T, 2)
            soft_connect_occupancy = soft_connect_occupancy.permute(1, 3, 0, 2).reshape(-1, B, T)
            soft_part_occupancy = torch.zeros([B, C, T, K]).to(points.device)
            soft_part_occupancy[:, dimC, :, self.connect] = soft_connect_occupancy
            if self.softmax_out:
                soft_part_occupancy = self.masked_softmax(soft_part_occupancy, soft_part_occupancy>0.0, dim=1, sigma=self.sigma)
            else:
                soft_part_occupancy = soft_part_occupancy.max(dim=1).values
            soft_part_occupancy = soft_part_occupancy.permute(0,2,1) * inside_bbox
            part_pred.update({'soft_part_occupancy': soft_part_occupancy})

            if not self.training:
                if self.softmax_out:
                    soft_occupancy = self.masked_softmax(soft_part_occupancy, soft_part_occupancy>0.0, dim=1, sigma=self.sigma)
                else:
                    soft_occupancy = soft_part_occupancy.max(dim=1).values  # (B, T)
                part_pred.update({'soft_occupancy': soft_occupancy})
        # t8 = time.time()
        # print('att code:', (t2-t1)*1000)
        # print('global pts:', (t3-t2)*1000)
        # print('canonical pts:', (t4-t3)*1000)
        # print('local feat:', (t5-t4)*1000)
        # print('decode:', (t6-t5)*1000)
        # print('re arrange:', (t7-t6)*1000)
        # print('soft:', (t8-t7)*1000)
        if ret_intermediate:
            return occupancy, part_pred

        return occupancy

    def collision_loss(self, point_cloud, smpl_output, ret_collision_mask=False):
        """
        Args:
            point_cloud (list or torch.tensor): B, N, 3
        """
        occupancy = self.query(point_cloud, smpl_output)
        loss = torch.relu(occupancy - self.level_set)  # B,N
        if ret_collision_mask is not None:
            inds = loss > 0
            return loss.sum(-1), inds
        return loss.sum(-1)  # B, N, 3

    @staticmethod
    def batchify_smpl_output(smpl_output):
        b_smpl_output_list = []
        batch_size = smpl_output.vertices.shape[0]
        for b_ind in range(batch_size):
            b_smpl_output_list.append(copy.copy(smpl_output))
            for key in b_smpl_output_list[-1].keys():
                val = getattr(smpl_output, key)
                if torch.is_tensor(val):
                    val = val[b_ind:b_ind+1].clone()
                setattr(b_smpl_output_list[-1], key, val)
        return b_smpl_output_list

    @torch.no_grad()
    def extract_mesh(self, smpl_output, grid_res=64, max_queries=100000, use_mise=False, mise_resolution0=32, mise_depth=3, get_vertex_colors=True, level=None, soft=False):
        scale = 1.1  # padding

        # act = lambda x: torch.log(x/(1-x+1e-6)+1e-6)  # revert sigmoid
        # level = 0.0
        if level is None:
            level = self.level_set
        occ_list = []

        verts = smpl_output.vertices
        B = verts.shape[0]
        device = verts.device
        part_colors = self.get_part_colors()
        b_smpl_output_list = self.batchify_smpl_output(smpl_output)

        b_min, b_max = verts.min(dim=1).values, verts.max(dim=1).values  # B,3
        gt_center = ((b_min + b_max)*0.5).cpu()  # B,3
        gt_scale = (b_max - b_min).max(dim=-1, keepdim=True).values.cpu()  # (B,1)
        gt_scale_gpu, gt_center_gpu = gt_scale.to(device), gt_center.to(device)

        # query grid
        if use_mise:
            from leap.tools.libmise import MISE
            value_grid_list = []
            for b_ind in range(B):
                mesh_extractor = MISE(mise_resolution0, mise_depth, level)
                points = mesh_extractor.query()
                while points.shape[0] != 0:
                    grid3d = torch.FloatTensor(points).to(device)
                    grid3d = scale*(grid3d/mesh_extractor.resolution - 0.5).reshape(1, -1, 3)  # [-0.5, 0.5]*scale
                    grid3d = grid3d*gt_scale_gpu[b_ind].reshape(1, 1, 1) + gt_center_gpu[b_ind].reshape(1, 1, 3)

                    # check occupancy for sampled points
                    occ_hats = []
                    for pts in torch.split(grid3d, max_queries, dim=1):
                        if not soft or self.soft_act is None:
                            occ_hats.append(self.query(pts.to(device=device), b_smpl_output_list[b_ind]).cpu().squeeze(0))  # N
                        else:
                            res = self.query(pts.to(device=device), b_smpl_output_list[b_ind], ret_intermediate=True)[1]
                            occ_hats.append(res['soft_occupancy'].cpu().squeeze(0))
                    values = torch.cat(occ_hats, dim=0).numpy().astype(np.float64)
                    # sample points again
                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()

                value_grid_list.append(mesh_extractor.to_dense())
            value_grid = np.stack(value_grid_list)
            grid_res = mesh_extractor.resolution
        else:
            grid3d = self.create_meshgrid3d(grid_res, grid_res, grid_res)  # range [0,G]
            grid3d = scale*(grid3d/grid_res - 0.5).reshape(1, -1, 3)  # (1,D*H*W,3) in range [-0.5, +0.5]*scale
            for grid_queries in torch.split(grid3d, max_queries//B, dim=1):
                pts = grid_queries.expand(B, -1, -1).to(device)*gt_scale_gpu.unsqueeze(1) + gt_center_gpu.unsqueeze(1)
                if not soft or self.soft_act is None:
                    occ_list.append(self.query(pts, smpl_output).cpu())  # B,N
                else:
                    res = self.query(pts, smpl_output, ret_intermediate=True)[1]
                    occ_hats.append(res['soft_occupancy'].cpu())
            value_grid = torch.cat(occ_list, dim=1).reshape(B, grid_res, grid_res, grid_res).numpy()  # B,D,H,W

        # extract meshes
        mesh_list = []
        for b_ind in range(B):
            try:
                verts, faces, normals, values = measure.marching_cubes(volume=value_grid[b_ind], gradient_direction='ascent', level=level)
            except:
                verts, faces, normals, values = measure.marching_cubes(volume=value_grid[b_ind], gradient_direction='ascent', level=value_grid[b_ind].mean())
            # vertices to world space
            verts = scale*(verts/(grid_res-1) - 0.5)
            verts = verts*gt_scale[b_ind].item() + gt_center[b_ind].cpu().numpy()
            
            # color meshes
            if get_vertex_colors:
                vertex_colors = self.color_points(torch.from_numpy(verts).reshape(1, -1, 3).to(device), b_smpl_output_list[b_ind], max_queries)[0]
            else:
                vertex_colors = None
            mesh_list.append(trimesh.Trimesh(verts, faces, vertex_colors=vertex_colors))

        return mesh_list

    @torch.no_grad()
    def color_points(self, pts, smpl_output=None, max_queries=100000):
        K = self.partitioner.num_parts
        part_colors = self.get_part_colors()
        B = pts.shape[0]
        part_colors = part_colors[:K+1]
        part_colors[-1, :] = 0  # set bg to black

        pts_colors = []
        for _v in torch.split(pts, max_queries//B, dim=1):
            part_pred = self.query(_v, smpl_output, ret_intermediate=True)[1]
            weight = (part_pred['part_occupancy'] / part_pred['part_occupancy'].sum(dim=1, keepdim=True))
            weight = weight.detach().cpu().numpy()
            pts_color = (part_colors[:K][None, :, None] * weight[..., None]).sum(axis=1).clip(0, 1)
            pts_colors.append(pts_color)

            # label = part_pred['part_occupancy'].argmax(dim=1)  # B,K,V -> B,V
            # label[part_pred['all_out']] = K  # all mlps say outside
            # inds = label.reshape(-1).cpu().numpy()
            # pts_colors.append(part_colors[inds].reshape((B, -1, 3)))

        pts_colors = np.concatenate(pts_colors, 1)
        return pts_colors

    @staticmethod
    def create_meshgrid3d(
        depth: int,
        height: int,
        width: int,
        device=torch.device('cpu'),
        dtype=torch.float32,
    ) -> torch.Tensor:
        """ Generate a coordinate grid in range [-0.5, 0.5].

        Args:
            depth (int): grid dim
            height (int): grid dim
            width (int): grid dim
        Return:
            grid tensor with shape :math:`(1, D, H, W, 3)`.
        """
        xs = torch.linspace(0, width, width, device=device, dtype=dtype)
        ys = torch.linspace(0, height, height, device=device, dtype=dtype)
        zs = torch.linspace(0, depth, depth, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid([xs, ys, zs], indexing='ij'), dim=-1).unsqueeze(0)  # 1xDxHxWx3

    @staticmethod
    def compute_iou(occ1, occ2, level=0.5):
        """ Computes the Intersection over Union (IoU) value for two sets of occupancy values.

        Args:
            occ1 (tensor): first set of occupancy values  (B, N)
            occ2 (tensor): second set of occupancy values  (B, N)
            level (float): threshold

        Returns:
            iou (tensor): mean IoU (scalar)
        """
        if occ1.dtype != torch.bool:
            occ1 = (occ1 >= level)
            occ2 = (occ2 >= level)

        # Compute IOU
        area_union = (occ1 | occ2).float().sum(axis=-1)
        area_intersect = (occ1 & occ2).float().sum(axis=-1)

        iou = (area_intersect / area_union.clamp_min(1))

        return iou.mean()

    @staticmethod
    def get_part_colors():
        return np.array([
            [ 8.94117647e-01,  1.01960784e-01,  1.09803922e-01],
            [ 2.15686275e-01,  4.94117647e-01,  7.21568627e-01],
            [ 3.01960784e-01,  6.86274510e-01,  2.90196078e-01],
            [ 5.96078431e-01,  3.05882353e-01,  6.39215686e-01],
            [ 1.00000000e+00,  4.98039216e-01,  3.58602037e-16],
            [ 1.00000000e+00,  1.00000000e+00,  2.00000000e-01],
            [ 6.50980392e-01,  3.37254902e-01,  1.56862745e-01],
            [ 9.68627451e-01,  5.05882353e-01,  7.49019608e-01],
            [ 6.00000000e-01,  6.00000000e-01,  6.00000000e-01],
            [ 1.00000000e+00,  9.17647059e-01,  8.47058824e-01],
            [ 4.94117647e-01,  3.58602037e-16,  1.84313725e-01],
            [ 7.92156863e-01,  7.29411765e-01,  3.72549020e-01],
            [ 5.25490196e-01,  4.78431373e-01,  3.58602037e-16],
            [ 9.29411765e-01,  6.50980392e-01,  5.76470588e-01],
            [ 5.05882353e-01,  4.03921569e-01,  4.15686275e-01],
            [ 7.80392157e-01,  3.56862745e-01,  4.43137255e-01],
            [ 6.86274510e-01,  5.37254902e-01,  3.52941176e-01],
            [ 6.27450980e-01,  8.23529412e-02,  0.00000000e+00],
            [ 1.00000000e+00,  4.35294118e-01,  3.80392157e-01],
            [ 7.37254902e-01,  7.17204074e-16,  2.82352941e-01],
            [ 1.00000000e+00,  9.09803922e-01,  5.52941176e-01],
            [ 1.00000000e+00,  7.84313725e-02,  4.15686275e-01],
            [ 7.72549020e-01,  7.52941176e-01,  6.66666667e-01],
            [ 5.68627451e-01,  2.62745098e-01,  2.98039216e-01],
            [ 0,  0,  0],
        ])
