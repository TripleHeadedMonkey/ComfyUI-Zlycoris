import torch
import torch.nn as nn
from toolkit.network_mixins import ToolkitNetworkMixin
from toolkit.models.loha import LoHaModule
import re
import weakref

class LoHaNetwork(ToolkitNetworkMixin, nn.Module):
    def __init__(
            self,
            text_encoder,
            unet,
            lora_dim=4,
            alpha=1,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            multiplier=1.0,
            train_unet=True,
            train_text_encoder=True,
            **kwargs
    ):
        nn.Module.__init__(self)
        super().__init__(
            text_encoder=text_encoder,
            unet=unet,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            **kwargs
        )
        self.network_type = "loha"
        self.peft_format = None
        
        # Force None to bypass incompatible base mixin save hooks
        self.base_model_ref = None
        
        # FIX: Explicitly set is_pixart. 
        # The base Mixin uses this in load_weights but doesn't initialize it itself.
        self.is_pixart = kwargs.get('is_pixart', False)
        
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self._multiplier = multiplier
        
        # Use ModuleList so PyTorch registers the parameters for saving
        self.loha_modules = nn.ModuleList()
        
        if train_unet:
            self.create_modules(unet, "lora_unet")
        if train_text_encoder:
            if isinstance(text_encoder, list):
                for i, te in enumerate(text_encoder):
                    self.create_modules(te, f"lora_te_{i+1}")
            else:
                self.create_modules(text_encoder, "lora_te")
                
    def create_modules(self, root_module, prefix):
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                if not module.weight.requires_grad and "refiner" not in prefix:
                     pass
                
                lora_name = f"{prefix}_{name}".replace('.', '_')
                
                loha_mod = LoHaModule(
                    lora_name=lora_name,
                    network=self,
                    org_module=module,
                    multiplier=self._multiplier,
                    lora_dim=self.lora_dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    rank_dropout=self.rank_dropout,
                    module_dropout=self.module_dropout,
                )
                
                # Append works the same way with ModuleList
                self.loha_modules.append(loha_mod)
                
                # Direct injection
                module.forward = loha_mod.forward
                module.loha_module = loha_mod

    def apply_to(self, text_encoder, unet, train_text_encoder, train_unet):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        for module in self.get_all_modules():
            for param in module.parameters():
                param.requires_grad = True

    def get_all_modules(self):
        return self.loha_modules

    def save_weights(self, file, dtype=torch.float16, metadata=None, extra_state_dict=None):
        if metadata is None:
            metadata = {}
        
        metadata["ss_network_module"] = "lycoris.kohya"
        metadata["ss_network_dim"] = str(self.lora_dim)
        metadata["ss_network_alpha"] = str(self.alpha)
        metadata["ss_network_args"] = str({'algo': 'loha'})
        
        super().save_weights(file, dtype, metadata, extra_state_dict)

    def prepare_optimizer_params(self, text_encoder_lr=None, unet_lr=None, default_lr=1e-4):
        all_params = []
        
        def enumerate_params(modules, lr):
            params = []
            for mod in modules:
                for param in mod.parameters():
                    if param.requires_grad:
                        params.append(param)
            return {"params": params, "lr": lr}

        if self.train_unet:
             unet_mods = [m for m in self.loha_modules if "lora_unet" in m.lora_name]
             all_params.append(enumerate_params(unet_mods, unet_lr or default_lr))
             
        if self.train_text_encoder:
             te_mods = [m for m in self.loha_modules if "lora_te" in m.lora_name]
             all_params.append(enumerate_params(te_mods, text_encoder_lr or default_lr))
             
        return all_params

    @torch.no_grad()
    def _update_torch_multiplier(self):
        multiplier = self._multiplier
        try:
            first_module = self.get_all_modules()[0]
        except IndexError:
            return 
        
        if hasattr(first_module, 'hada_w1_a'):
            device = first_module.hada_w1_a.device
            dtype = first_module.hada_w1_a.dtype
            if hasattr(first_module.hada_w1_a, '_memory_management_device'):
                device = first_module.hada_w1_a._memory_management_device
        else:
            raise ValueError(f"Unknown module type: {type(first_module)}")

        with torch.no_grad():
            tensor_multiplier = None
            if isinstance(multiplier, int) or isinstance(multiplier, float):
                # Create scalar tensor
                tensor_multiplier = torch.tensor(multiplier).to(device, dtype=dtype)
            elif isinstance(multiplier, list):
                tensor_multiplier = torch.tensor(multiplier).to(device, dtype=dtype)
            elif isinstance(multiplier, torch.Tensor):
                tensor_multiplier = multiplier.clone().detach().to(device, dtype=dtype)

            self.torch_multiplier = tensor_multiplier.clone().detach()