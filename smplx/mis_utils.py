# coding: UTF-8

"""
    @date:  2022.08.10  week33  星期3
    @func:  不同于原始的flame的edge_subdivide, 对MANO, 去掉uv相关的部分.
    @ref:   https://github.com/philgras/neural-head-avatars/blob/0048afe9c9034157c63838801e9a2dd3126f806e/nha/util/meshes.py#L17
"""
import torch
import numpy as np
import torch.nn.functional as F


def append_edge(edge_map_, edges_, idx_a, idx_b):
    if idx_b < idx_a:
        idx_a, idx_b = idx_b, idx_a

    if not (idx_a, idx_b) in edge_map_:
        e_id = len(edges_)
        edges_.append([idx_a, idx_b])
        edge_map_[(idx_a, idx_b)] = e_id
        edge_map_[(idx_b, idx_a)] = e_id


def edge_subdivide(vertices, faces):
    """
    subdivides mesh based on edge midpoints. every triangle is subdivided into 4 child triangles.
    old faces are kept in array
    :param vertices: V x 3 ... vertex coordinates
    :param faces: F x 3 face vertex idx array
    :return:
        - vertices ... np.array of vertex coordinates with shape V + n_edges x 3
        - faces ... np.array of face vertex idcs with shape F + 4*F x 3
        - edges ... np.array of shape n_edges x 2 giving the indices of the vertices of each edge
        all returns are a concatenation like np.concatenate((array_old, array_new), axis=0) so that
        order of old entries is not changed and so that also old faces are still present.
    """
    n_faces = faces.shape[0]
    n_vertices = vertices.shape[0]

    # if self.edges is None:
    # if True:
    # compute edges
    edges = []
    edge_map = dict()
    for i in range(0, n_faces):
        append_edge(edge_map, edges, faces[i, 0], faces[i, 1])
        append_edge(edge_map, edges, faces[i, 1], faces[i, 2])
        append_edge(edge_map, edges, faces[i, 2], faces[i, 0])
    n_edges = len(edges)
    edges = np.array(edges).astype(int)

    #    print('edges:', edges.shape)
    #    print('self.edge_map :', len(edge_map ))
    #
    #    print('vertices:', vertices.shape)
    #    print('faces:', faces.shape)

    ############
    # vertices
    v = np.zeros((n_vertices + n_edges, 3))
    # copy original vertices
    v[:n_vertices, :] = vertices
    # compute edge midpoints
    vertices_edges = vertices[edges]
    # edge_len = (vertices_edges[:,0]-vertices_edges[:,1]).norm(dim=-1)
    # valid = edge_len > 0.002
    v[n_vertices:, :] = (0.5 * (vertices_edges[:, 0] + vertices_edges[:, 1]))

    # new topology
    f = np.concatenate((faces, np.zeros((4 * n_faces, 3))), axis=0)
    # f_uv = np.zeros((4*n_faces*3, 2))
    for i in range(0, n_faces):
        # vertex ids
        a = int(faces[i, 0])
        b = int(faces[i, 1])
        c = int(faces[i, 2])
        ab = n_vertices + edge_map[(a, b)]
        bc = n_vertices + edge_map[(b, c)]
        ca = n_vertices + edge_map[(c, a)]

        ## triangle 1
        f[n_faces + 4 * i, 0] = a
        f[n_faces + 4 * i, 1] = ab
        f[n_faces + 4 * i, 2] = ca

        ## triangle 2
        f[n_faces + 4 * i + 1, 0] = ab
        f[n_faces + 4 * i + 1, 1] = b
        f[n_faces + 4 * i + 1, 2] = bc

        ## triangle 3
        f[n_faces + 4 * i + 2, 0] = ca
        f[n_faces + 4 * i + 2, 1] = ab
        f[n_faces + 4 * i + 2, 2] = bc

        ## triangle 4
        f[n_faces + 4 * i + 3, 0] = ca
        f[n_faces + 4 * i + 3, 1] = bc
        f[n_faces + 4 * i + 3, 2] = c

    return v, f[n_faces:], edges


"""
code heavily inspired from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/sample_points_from_meshes.html
just changed functionality such that only sampled face idcs and barycentric coordinates are returned. User 
has to do the rest 
"""


def face_vertices(vertices, faces):
    """
    :param vertices: [x size, number of vertices, 3]
    :param faces: [x size, number of faces, 3]
    :return: [x size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(
        0,
        faces[:, 1].long(),
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 2].long(),
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 0].long(),
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
        ),
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


import torch
import os
import numpy as np
import openmesh as om


def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    from generate_spiral_seq import extract_spirals
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals


# 封口.
# 2022.08.17
def seal(verts, faces, left=False):
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (verts[circle_v_id, :]).mean(0)

    verts = np.vstack([verts, center])
    center_v_id = verts.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        if left:
            new_faces = [circle_v_id[i-1], center_v_id, circle_v_id[i]]
        else:
            new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id]
        faces = np.vstack([faces, new_faces])
    return verts, faces
