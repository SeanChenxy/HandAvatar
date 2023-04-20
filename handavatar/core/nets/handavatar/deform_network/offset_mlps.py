import torch
from handavatar.core.nets.hand_nerf.deform_network.siren_mlps import SIRENMLP
from smplx.lbs import batch_rodrigues


class DeformNetwork(torch.nn.Module):
    def __init__(self, num_verts, cond_feats=0, hidden_feats=128, hidden_layers=6, **_):
        super().__init__()
        self._vert_feats = torch.nn.Parameter(torch.zeros(1, num_verts, 32), requires_grad=True)
        self._vert_feats.data.normal_(0.0, 0.02)
        self._SIREN = SIRENMLP(input_dim=35,
                                output_dim=3,
                                condition_dim=cond_feats,
                                hidden_dim=hidden_feats,
                                hidden_layers=hidden_layers)

    def forward(self, x, conditions):
        conditions = conditions.reshape(-1, 3)
        conditions = batch_rodrigues(conditions).view(-1, 135)
        x = torch.cat([x, self._vert_feats], dim=-1)
        res = self._SIREN.forward(vertices=x, additional_conditioning=conditions)
        return torch.tanh(res)
