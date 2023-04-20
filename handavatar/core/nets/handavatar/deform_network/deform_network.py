import torch
import torch.nn as nn
from core.nets.handavatar.component_factory import \
    load_positional_embedder


class DeformNetwork(nn.Module):
    def __init__(self, d_in, d_out, dims, feature_vector_size, multires, num_verts, verts=None, embedder=None, **kwargs):
        super(DeformNetwork, self).__init__()

        dims = [d_in+feature_vector_size,] + dims + [d_out,]
        self.feature_vector_size=feature_vector_size
        self.embed_fn = None
        self.multires = multires
        if d_in==3 and multires > 0 and embedder is not None:
            get_embedder = load_positional_embedder(embedder)
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch + feature_vector_size
            
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if l==self.num_layers-2:
                torch.nn.init.normal_(lin.weight, mean=0., std=0.001)
                torch.nn.init.constant_(lin.bias, 0.)
            setattr(self, "lin" + str(l), lin)

        if d_in==3:
            self.register_buffer('vert_dict', verts)
            if self.embed_fn is not None:
                self.vert_dict = self.embed_fn(self.vert_dict)
        else:
            self.vert_dict = torch.nn.Parameter(torch.zeros(1, num_verts, d_in), requires_grad=True)
            self.vert_dict.data.normal_(0.0, 0.02)

        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def forward(self, conds, **kwargs):
        
        # if self.embed_fn is not None:
        #     ps = self.embed_fn(ps)

        x = torch.cat([self.vert_dict.expand(conds.shape[0], -1, -1), 
            conds.view(-1, 1, self.feature_vector_size).expand(-1, self.vert_dict.shape[1], self.feature_vector_size)], 
            dim=-1).view(-1, self.vert_dict.shape[-1] + self.feature_vector_size)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        return x.view(conds.shape[0], self.vert_dict.shape[1], 3)