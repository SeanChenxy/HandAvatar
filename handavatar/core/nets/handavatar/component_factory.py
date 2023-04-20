import imp

def load_positional_embedder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).get_embedder

def load_canonical_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).CanonicalMLP

def load_mweight_vol_decoder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).MotionWeightVolumeDecoder

def load_pose_decoder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).BodyPoseRefiner

def load_non_rigid_motion_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).NonRigidMotionMLP

def load_implicit_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).ImplicitNetwork

def load_render_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).RenderNetwork

def load_deform_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).DeformNetwork

def load_s_network(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).SingleVarianceNetwork

def load_shadow_network(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).ShadowNetwork