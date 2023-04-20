# coding: UTF-8
"""
    @date:  2022.07.22  week30  星期五
    @func:  RenderingNetwork
"""

import torch
import torch.nn as nn
from core.nets.handavatar.embedders.log_fourier import get_embedder


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 multires_pos=0,
                 multires_normal=0,
                 squeeze_out=True,
                 **_):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view, input_dims=3)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
        
        self.embedpos_fn = None
        if multires_pos > 0:
            embedpos_fn, input_ch = get_embedder(multires_pos, input_dims=3)
            self.embedpos_fn = embedpos_fn
            dims[0] += (input_ch - 3)
        
        self.embednormal_fn = None
        if multires_normal > 0:
            embednormal_fn, input_ch = get_embedder(multires_pos, input_dims=3)
            self.embednormal_fn = embednormal_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        
        # import pdb; pdb.set_trace()
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        if self.embedpos_fn is not None:
            points = self.embedpos_fn(points)
        if self.embednormal_fn is not None:
            normals = self.embednormal_fn(normals)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        elif self.mode == 'no_feat':
            rendering_input = torch.cat([points, view_dirs, normals], dim=-1)
        elif self.mode == 'pos_dir':
            rendering_input = torch.cat([points, view_dirs], dim=-1)
        elif self.mode == 'no_pts':
            rendering_input = torch.cat([view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'feat':
            rendering_input = feature_vectors

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x