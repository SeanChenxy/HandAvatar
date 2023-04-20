import torch
import smplx
import imp

def attach_pairof(parametric_body: smplx.SMPL, smpl_cfg: dict, device=None):
    model_name = smpl_cfg['pairof_name']
    module_path = model_name.replace(".", "/") + ".py"
    model = imp.load_source(model_name, module_path).COAPBodyModel
    coap_body = model(parametric_body, smpl_cfg)
    setattr(parametric_body, 'coap', coap_body)
    if device is not None:
        parametric_body = parametric_body.to(device=device)

    # overwrite smpl functions
    def reset_params(self, **params_dict) -> None:
        with torch.no_grad():
            for param_name, param in self.named_parameters():
                if 'coap' in param_name:  # disable reset of coap parameters
                    continue
                if param_name in params_dict:
                    param[:] = torch.tensor(params_dict[param_name])
                else:
                    param.fill_(0)
    setattr(parametric_body, 'reset_params', lambda **x: reset_params(parametric_body, **x))
    
    return parametric_body


__all__ = [
    attach_pairof,
]
