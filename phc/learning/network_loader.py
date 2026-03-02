import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from phc.utils import torch_utils

from easydict import EasyDict as edict


def load_mlp(loading_keys, checkpoint, actvation_func):
    
    loading_keys_linear = [k for k in loading_keys if k.endswith('weight')]
    nn_modules = []
    for idx, key in enumerate(loading_keys_linear):
        if len(checkpoint['model'][key].shape) == 1: # layernorm
            layer = torch.nn.LayerNorm(*checkpoint['model'][key].shape[::-1])
            nn_modules.append(layer)
        elif len(checkpoint['model'][key].shape) == 2: # nn
            layer = nn.Linear(*checkpoint['model'][key].shape[::-1])
            nn_modules.append(layer)
            if idx < len(loading_keys_linear) - 1:
                nn_modules.append(actvation_func())
        else:
            raise NotImplementedError
        
    net = nn.Sequential(*nn_modules)
    
    state_dict = net.state_dict()
    
    for idx, key_affix in enumerate(state_dict.keys()):
        state_dict[key_affix].copy_(checkpoint['model'][loading_keys[idx]])
        
    for param in net.parameters():
        param.requires_grad = False
        
    return net



def load_combat_prior(checkpoint, device="cpu", activation="silu"):
    act_fn = torch_utils.activation_facotry(activation)

    # 1) Collect all _combat_prior_mlp parameter keys (keep original checkpoint order)
    mlp_prefix = "a2c_network._combat_prior_mlp"
    prior_loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(mlp_prefix)]

    # 2) Append final output layer _combat_prior_output weight/bias
    prior_loading_keys += [
        "a2c_network._combat_prior_output.weight",
        "a2c_network._combat_prior_output.bias",
    ]

    # 3) Rebuild with the existing constructor in key order and copy weights
    prior_decoder = load_mlp(prior_loading_keys, checkpoint, act_fn)

    prior_decoder.to(device)
    prior_decoder.eval()
    for p in prior_decoder.parameters():
        p.requires_grad = False

    return prior_decoder

