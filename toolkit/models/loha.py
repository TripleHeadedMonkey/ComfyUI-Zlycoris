import torch
import torch.nn as nn
import torch.nn.functional as F
from toolkit.network_mixins import ToolkitModuleMixin

class LoHaModule(ToolkitModuleMixin, nn.Module):
    def __init__(
            self, 
            lora_name, 
            network, 
            org_module: nn.Module, 
            multiplier=1.0, 
            lora_dim=4, 
            alpha=1, 
            dropout=0.0,
            rank_dropout=0.0, 
            module_dropout=0.0,
            use_cp=False, 
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(network=network)
        
        self.lora_name = lora_name
        self.org_module = [org_module] 
        # Capture original forward to avoid recursion
        self.org_forward = org_module.forward
        
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self._multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        
        if isinstance(org_module, nn.Conv2d):
            self.is_conv = True
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.down_shape = (lora_dim, in_dim, k_size[0], k_size[1])
            self.up_shape = (out_dim, lora_dim, 1, 1)
        else:
            self.is_conv = False
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.down_shape = (lora_dim, in_dim)
            self.up_shape = (out_dim, lora_dim)

        self.hada_w1_a = nn.Parameter(torch.empty(self.down_shape))
        self.hada_w1_b = nn.Parameter(torch.empty(self.up_shape))
        self.hada_w2_a = nn.Parameter(torch.empty(self.down_shape))
        self.hada_w2_b = nn.Parameter(torch.empty(self.up_shape))

        self.scale = alpha / lora_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.hada_w1_a, std=0.1)
        nn.init.normal_(self.hada_w1_b, std=0.1)
        nn.init.normal_(self.hada_w2_a, std=0.1)
        nn.init.constant_(self.hada_w2_b, 0)

    def get_diff_weight(self):
        if self.is_conv:
            w1 = (self.hada_w1_b.flatten(start_dim=1) @ self.hada_w1_a.flatten(start_dim=1)).view(
                self.hada_w1_b.shape[0], self.hada_w1_a.shape[1], self.hada_w1_a.shape[2], self.hada_w1_a.shape[3]
            )
            w2 = (self.hada_w2_b.flatten(start_dim=1) @ self.hada_w2_a.flatten(start_dim=1)).view(
                self.hada_w2_b.shape[0], self.hada_w2_a.shape[1], self.hada_w2_a.shape[2], self.hada_w2_a.shape[3]
            )
        else:
            w1 = self.hada_w1_b @ self.hada_w1_a
            w2 = self.hada_w2_b @ self.hada_w2_a
            
        return (w1 * w2) * self.scale

    def forward(self, x, *args, **kwargs):
        network = self.network_ref()
        if not network.is_active or network.is_merged_in:
            return self.org_forward(x, *args, **kwargs)

        org_out = self.org_forward(x, *args, **kwargs)
        
        diff_weight = self.get_diff_weight()
        
        # 1. Sync diff_weight dtype with input (Fixes BFloat16 mismatches)
        if diff_weight.dtype != x.dtype:
            diff_weight = diff_weight.to(dtype=x.dtype)
        
        # 2. Robust Multiplier Handling
        multiplier = network.multiplier if network.multiplier is not None else self._multiplier
        
        # Handle List (Unwrap if possible)
        if isinstance(multiplier, list):
            if len(multiplier) == 1:
                multiplier = multiplier[0]
            # If len > 1, it's a vector, keep as list for now, will become Tensor below
                
        # Handle Tensor (Convert scalars to Python float)
        if isinstance(multiplier, torch.Tensor):
            if multiplier.numel() == 1:
                multiplier = multiplier.item()
            elif multiplier.dtype != diff_weight.dtype:
                multiplier = multiplier.to(dtype=diff_weight.dtype, device=diff_weight.device)
        
        # 3. Apply Multiplier to WEIGHTS
        # Using pure float multiplication avoids PyTorch "Integer Tensor" confusion
        diff_weight = diff_weight * multiplier
            
        if self.is_conv:
             out_diff = F.conv2d(
                 x, 
                 diff_weight, 
                 bias=None, 
                 stride=self.org_module[0].stride, 
                 padding=self.org_module[0].padding, 
                 dilation=self.org_module[0].dilation, 
                 groups=self.org_module[0].groups
             )
        else:
            out_diff = F.linear(x, diff_weight)
            
        return org_out + out_diff

    def merge_in(self, merge_weight=1.0):
        if self.network_ref().is_merged_in:
            return
        
        with torch.no_grad():
            weight = self.org_module[0].weight
            diff = self.get_diff_weight() * merge_weight
            weight.add_(diff.to(weight.device))

    def merge_out(self, merge_weight=1.0):
        if not self.network_ref().is_merged_in:
            return
            
        with torch.no_grad():
            weight = self.org_module[0].weight
            diff = self.get_diff_weight() * merge_weight
            weight.sub_(diff.to(weight.device))
            
    def parameters(self, recurse: bool = True):
        return [self.hada_w1_a, self.hada_w1_b, self.hada_w2_a, self.hada_w2_b]