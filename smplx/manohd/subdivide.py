import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import smplx
import trimesh
from smplx.mis_utils import edge_subdivide
import torch
import numpy as np
import json
import openmesh as om
import pickle
from sklearn.neighbors import KDTree


def _next_ring(mesh, last_ring, other):
    res = []

    def is_new_vertex(idx):
        return (idx not in last_ring and idx not in other and idx not in res)

    for vh1 in last_ring:
        vh1 = om.VertexHandle(vh1)
        after_last_ring = False
        for vh2 in mesh.vv(vh1):
            if after_last_ring:
                if is_new_vertex(vh2.idx()):
                    res.append(vh2.idx())
            if vh2.idx() in last_ring:
                after_last_ring = True
        for vh2 in mesh.vv(vh1):
            if vh2.idx() in last_ring:
                break
            if is_new_vertex(vh2.idx()):
                res.append(vh2.idx())
    return res

def extract_spirals(mesh, seq_length, dilation=1):
    # output: spirals.size() = [N, seq_length]
    spirals = []
    one_ring_list = []
    next_ring_list = []
    for vh0 in mesh.vertices():
        reference_one_ring = []
        for vh1 in mesh.vv(vh0):
            reference_one_ring.append(vh1.idx())
        spiral = [vh0.idx()]
        one_ring = list(reference_one_ring)
        one_ring_list.append(one_ring)

        last_ring = one_ring
        next_ring = _next_ring(mesh, one_ring, spiral)
        next_ring_list.append(next_ring)

        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(mesh, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            kdt = KDTree(mesh.points(), metric='euclidean')
            spiral = kdt.query(np.expand_dims(mesh.points()[spiral[0]],
                                              axis=0),
                               k=seq_length * dilation,
                               return_distance=False).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(spiral[:seq_length * dilation][::dilation])
    return one_ring_list, next_ring_list, spirals

def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    one_ring_list, next_ring_list, spirals = extract_spirals(mesh, seq_length=seq_length, dilation=dilation)
    return one_ring_list, next_ring_list, spirals

def smooth(v, f, start, times=1):

    one_ring_list, next_ring_list, spirals = preprocess_spiral(f, 9, v)
    v_org = v.copy()

    for _ in range(times):
        target_list = list(range(v.shape[0]))[0:]
        for idx in target_list:
            indices = one_ring_list[idx]
            # indices += [idx]
            mean = v_org[indices + [idx]].mean(axis=0)
            v[idx] = mean #(v[idx] - mean) * 0.7 + mean

    return v, spirals


def sub_mano(mano, t, pretrain=None):
    for i in range(t):
        n_verts = mano.v_template.shape[0]
        verts, faces, edges = edge_subdivide(vertices=mano.v_template, faces=mano.faces)
        if i==t-1:
            verts, spirals = smooth(verts, faces, n_verts, 3)

        print('upsample mano to ', verts.shape[0])
        subdivided_mesh = trimesh.Trimesh(verts, faces)
        # _ = subdivided_mesh.export(f'smplx/out/subdivided_template_{i}.obj')

        new_shapedirs = mano.shapedirs[edges]
        new_shapedirs = new_shapedirs.mean(dim=1)  # n_edges x 3 x 10
        shapedirs = torch.cat((mano.shapedirs, new_shapedirs), dim=0)

        new_posedirs = mano.posedirs.permute(1, 0).view(n_verts, 3, 135)  # V x 3 x 135
        new_posedirs = new_posedirs[edges]  # n_edges x 2 x 3 x 135
        new_posedirs = new_posedirs.mean(dim=1)  # n_edges x 3 x 135
        new_posedirs = new_posedirs.view(len(edges) * 3, 135).permute(1, 0)
        posedirs = torch.cat((mano.posedirs, new_posedirs), dim=1)

        # lbs & J_regressor
        new_J_regressor = torch.zeros(16, len(edges)).to(mano.J_regressor.dtype).to(mano.J_regressor.device)
        J_regressor = torch.cat((mano.J_regressor, new_J_regressor), dim=1)

        new_lbs_weights = mano.lbs_weights[edges]  # n_edges x 2 x 16
        new_lbs_weights = new_lbs_weights.mean(dim=1)  # n_edges x 16
        lbs_weights = torch.cat((mano.lbs_weights, new_lbs_weights), dim=0)

        mano.faces = faces.astype('int32')
        mano.faces_tensor = torch.from_numpy(mano.faces)
        mano.v_template = torch.from_numpy(verts).float()

        mano.posedirs = posedirs
        mano.shapedirs = shapedirs

        mano.J_regressor = J_regressor
        
        if i==0 and t>1 and pretrain is not None:
            pretrain_weight = f'smplx/out/{pretrain}/ckpts/lbs_weights.pth'
            lbs_weights = torch.load(pretrain_weight)
            print('load pretrain', pretrain_weight)
            
        mano.lbs_weights = lbs_weights

    mano.update_seal()
    return mano, edges, spirals
