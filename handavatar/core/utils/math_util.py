import numpy as np


def convert(rot, src, tar):
  eps = np.finfo(np.float32).eps
  if src == 'axangle':
    data_shape = rot.shape[:-1]
    rot = np.reshape(rot, [-1, 3])
    if tar == 'quat':
      rad = np.linalg.norm(rot, axis=-1, keepdims=True)
      ax = rot / np.maximum(rad, eps)
      w = np.cos(rad / 2)
      xyz = np.sin(rad / 2) * ax
      quat = np.concatenate([w, xyz], -1)
      quat = np.reshape(quat, data_shape + (4,))
      # quat_neg = quat * -1
      # quat = np.where(quat[:, 0:1] > 0, quat, quat_neg)
      return quat
    if tar == 'rotmat':
      theta = np.linalg.norm(rot, axis=-1, keepdims=True)
      c = np.cos(theta)
      s = np.sin(theta)
      t = 1 - c
      x, y, z = np.split(rot / np.maximum(theta, eps), 3, axis=-1)
      rotmat = np.stack([
        t*x*x + c, t*x*y - z*s, t*x*z + y*s,
        t*x*y + z*s, t*y*y + c, t*y*z - x*s,
        t*x*z - y*s, t*y*z + x*s, t*z*z + c
      ], 1)
      rotmat = np.reshape(rotmat, data_shape + (3, 3))
      return rotmat
  elif src == 'quat':
    data_shape = rot.shape[:-1]
    rot = np.reshape(rot, [-1, 4])
    if tar == 'rotmat':
      w, x, y, z = np.split(rot, 4, axis=-1)
      rotmat = np.stack([
        1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w,
        2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w,
        2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y
      ], -1)
      rotmat = np.reshape(rotmat, data_shape + (3, 3))
      return rotmat
    if tar == 'axangle':
      angle = 2 * np.arccos(rot[:, 0:1])
      axis = rot[:, 1:] / np.sqrt(1 - np.square(rot[:, 0:1]))
      axangle = axis * angle
      axangle = np.reshape(axangle, data_shape + (3,))
      return axangle
  elif src == 'rotmat':
    data_shape = rot.shape[:-2]
    rot = np.reshape(rot, [-1, 3, 3])
    if tar == 'rot6d':
      rot6d = np.reshape(np.transpose(rot[:, :, :2], [0, 2, 1]), data_shape + (6,))
      return rot6d
    if tar == 'axangle':
      # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
      angle = np.arccos((rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2] - 1) / 2)
      angle = np.expand_dims(angle, -1)
      norm = np.sqrt(
        np.square(rot[:, 2, 1] - rot[:, 1, 2]) + \
        np.square(rot[:, 0, 2] - rot[:, 2, 0]) + \
        np.square(rot[:, 1, 0] - rot[:, 0, 1])
      )
      norm = np.maximum(norm, np.finfo(np.float32).eps)
      x = (rot[:, 2, 1] - rot[:, 1, 2]) / norm
      y = (rot[:, 0, 2] - rot[:, 2, 0]) / norm
      z = (rot[:, 1, 0] - rot[:, 0, 1]) / norm
      axangle = np.stack([x, y, z], -1) * angle
      axangle = np.reshape(axangle, data_shape + (3,))
      return axangle
    if tar == 'quat':
      # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
      quat = []
      for i in range(rot.shape[0]):
        tr = rot[i, 0, 0] + rot[i, 1, 1] + rot[i, 2, 2]
        if tr > 0:
          S = np.sqrt(tr+1.0) * 2
          qw = S / 4
          qx = (rot[i, 2, 1] - rot[i, 1, 2]) / S
          qy = (rot[i, 0, 2] - rot[i, 2, 0]) / S
          qz = (rot[i, 1, 0] - rot[i, 0, 1]) / S
        elif rot[i, 0, 0] > rot[i, 1, 1] and rot[i, 0, 0] > rot[i, 2, 2]:
          S = np.sqrt(1.0 + rot[i, 0, 0] - rot[i, 1, 1] - rot[i, 2, 2]) * 2
          qw = (rot[i, 2, 1] - rot[i, 1, 2]) / S
          qx = S / 4
          qy = (rot[i, 0, 1] + rot[i, 1, 0]) / S
          qz = (rot[i, 0, 2] + rot[i, 2, 0]) / S
        elif rot[i, 1, 1] > rot[i, 2, 2]:
          S = np.sqrt(1.0 + rot[i, 1, 1] - rot[i, 0, 0] - rot[i, 2, 2]) * 2
          qw = (rot[i, 0, 2] - rot[i, 2, 0]) / S
          qx = (rot[i, 0, 1] + rot[i, 1, 0]) / S
          qy = S / 4
          qz = (rot[i, 1, 2] + rot[i, 2, 1]) / S
        else:
          S = np.sqrt(1.0 + rot[i, 2, 2] - rot[i, 0, 0] - rot[i, 1, 1]) * 2
          qw = (rot[i, 1, 0] - rot[i, 0, 1]) / S
          qx = (rot[i, 0, 2] + rot[i, 2, 0]) / S
          qy = (rot[i, 1, 2] + rot[i, 2, 1]) / S
          qz = S / 4
        quat.append([qw, qx, qy, qz])
      quat = np.array(quat)
      quat = np.reshape(quat, data_shape + (4,))
      return quat
  elif src == 'rot6d':
    data_shape = rot.shape[:-1]
    rot = np.reshape(rot, [-1, 6])
    if tar == 'rotmat':
      col0 = rot[:, 0:3] / \
          np.maximum(np.linalg.norm(rot[:, 0:3], axis=-1, keepdims=True), eps)
      col1 = rot[:, 3:6] - np.sum((col0 * rot[:, 3:6]), axis=-1, keepdims=True) * col0
      col1 = col1 / np.maximum(np.linalg.norm(col1, axis=-1, keepdims=True), eps)
      col2 = np.cross(col0, col1)
      rotmat = np.stack([col0, col1, col2], -1)
      rotmat = np.reshape(rotmat, data_shape + (3, 3))
      return rotmat
    if tar == 'axangle':
      rotmat = convert(rot, 'rot6d', 'rotmat')
      axangle = convert(rotmat, 'rotmat', 'axangle')
      axangle = np.reshape(axangle, data_shape + (3,))
      return axangle

  raise NotImplementedError(f'Unsupported conversion: from {src} to {tar}.')


def sphere_sampling(n):
  theta = np.random.uniform(0, 2 * np.pi, size=n)
  phi = np.random.uniform(0, np.pi, size=n)
  v = np.stack(
    [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1
  )
  return v


def random_rotation(n):
  axis = sphere_sampling(n)
  angle = np.random.uniform(np.pi, size=[n, 1])
  return convert(axis * angle, 'axangle', 'rotmat')


def slerp_batch(a, b, t):
  # x = a * (1 - t) + b * t
  dot = np.einsum('NJD, NJD -> NJ', a, b)
  omega = np.expand_dims(np.arccos(np.clip(dot, 0, 1), dtype=np.float32), -1)
  so = np.sin(omega, dtype=np.float32)
  so[so == 0] = np.finfo(np.float32).eps
  p = np.sin((1 - t) * omega, dtype=np.float32) / so * a + \
      np.sin(t * omega, dtype=np.float32) / so * b
  mask = np.tile(np.prod(a == b, axis=-1, keepdims=True, dtype=np.bool), (1, 4))
  p = np.where(mask, a, p)
  return p


def rotmat_rel_to_abs_batch(rel_rotmat, parents):
  rel_rotmat = np.array(rel_rotmat)
  n_joints = len(parents)
  abs_rotmat = [None] * n_joints
  for c in range(n_joints):
    abs_rotmat[c] = rel_rotmat[:, c]
    p = parents[c]
    while p is not None:
      abs_rotmat[c] = np.einsum('nhw, nwk -> nhk', rel_rotmat[:, p], abs_rotmat[c])
      p = parents[p]
  abs_rotmat = np.stack(abs_rotmat, 1)
  return abs_rotmat


def rotmat_abs_to_rel(abs_rotmat, parents):
  n_joints = len(parents)
  rel_rotmat = [None] * n_joints
  for c in range(n_joints):
    p = parents[c]
    if p is None:
      rel_rotmat[c] = abs_rotmat[c]
    else:
      rel_rotmat[c] = np.dot(abs_rotmat[p].T, abs_rotmat[c])
  rel_rotmat = np.stack(rel_rotmat, 0)
  return rel_rotmat


def rotmat_rel_to_abs(rel_rotmat, parents, batch=False):
  if not batch:
    rel_rotmat = np.expand_dims(rel_rotmat, 0)
  n_joints = len(parents)
  abs_rotmat = [None] * n_joints
  for c in range(n_joints):
    abs_rotmat[c] = rel_rotmat[:, c]
    p = parents[c]
    while p is not None:
      abs_rotmat[c] = \
        np.einsum('nhw, nwk -> nhk', rel_rotmat[:, p], abs_rotmat[c])
      p = parents[p]
  abs_rotmat = np.stack(abs_rotmat, 1)
  if not batch:
    abs_rotmat = abs_rotmat[0]
  return abs_rotmat


def keypoints_to_bones_batch(keypoints, parents):
  keypoints = np.array(keypoints)
  bones = []
  for c, p in enumerate(parents):
    if p is None:
      bones.append(keypoints[:, c])
    else:
      bones.append(keypoints[:, c] - keypoints[:, p])
  bones = np.stack(bones, 1)
  return bones


def bones_to_keypoints_batch(bones, parents):
  bones = np.array(bones)
  keypoints = []
  for c, p in enumerate(parents):
    if p is None:
      keypoints.append(bones[:, c])
    else:
      keypoints.append(bones[:, c] + keypoints[p])
  if type(bones) != list:
    keypoints = np.stack(keypoints, 1)
  return keypoints


def forward_kinematics_batch(ref_bones, abs_rotmat, parents):
  ref_bones = np.array(ref_bones)
  abs_rotmat = np.array(abs_rotmat)
  bones = np.einsum('NJHW, NJW -> NJH', abs_rotmat, ref_bones)
  keypoints = bones_to_keypoints_batch(bones, parents)
  return keypoints, bones


def measure_hand_size(keypoints, skeleton):
  bones = keypoints_to_bones_batch([keypoints], skeleton.parents)[0]
  bones = np.array([[bones[skeleton.labels.index(f + k)] for f in 'IMR'] for k in '0123'])
  return np.mean(np.linalg.norm(bones, axis=-1))
