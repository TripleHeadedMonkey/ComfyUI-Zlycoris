import os
import types
from typing import List, Optional

import huggingface_hub
import torch
import torch.utils.checkpoint
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager
from safetensors.torch import load_file

from transformers import AutoTokenizer, Qwen3ForCausalLM
from diffusers import AutoencoderKL

try:
    from diffusers import ZImagePipeline
    from diffusers.models.transformers import ZImageTransformer2DModel
except ImportError:
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version."
    )


scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}

# --- SURGICAL TOOL: MONKEY PATCH ---
# Prevents AI Toolkit from crashing when it tries to set requires_grad=True 
# on quantized (integer) weights.
def safe_requires_grad_(self, requires_grad=True):
    for param in self.parameters():
        if param.dtype.is_floating_point:
            param.requires_grad = requires_grad
    return self

class ZImageModel(BaseModel):
    arch = "zimage"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["ZImageTransformer2DModel", "Qwen3ForCausalLM", "Qwen2ForCausalLM"]

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2

    def load_training_adapter(self, transformer: ZImageTransformer2DModel):
        self.print_and_status_update("Loading assistant LoRA")
        lora_path = self.model_config.assistant_lora_path
        if not os.path.exists(lora_path):
            lora_splits = lora_path.split("/")
            if len(lora_splits) != 3:
                raise ValueError(f"Invalid LoRA path: {lora_path}")
            repo_id = "/".join(lora_splits[:2])
            filename = lora_splits[2]
            try:
                lora_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
                self.model_config.assistant_lora_path = lora_path
            except Exception as e:
                raise ValueError(f"Failed to download assistant LoRA: {e}")

        lora_state_dict = load_file(lora_path)
        dim = int(lora_state_dict["diffusion_model.layers.0.attention.to_k.lora_A.weight"].shape[0])

        new_sd = {}
        for key, value in lora_state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        lora_state_dict = new_sd

        network_config = NetworkConfig(type="lora", linear=dim, linear_alpha=dim, transformer_only=True)
        
        LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=self.target_lora_modules,
            is_assistant_adapter=True,
            is_ara=True,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        self.print_and_status_update("Merging in assistant LoRA")
        
        network.force_to(transformer.device, dtype=self.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)
        network.merge_in(merge_weight=1.0)
        network.is_merged_in = False
        self.assistant_lora = network
        self.assistant_lora.multiplier = -1.0
        self.assistant_lora.is_active = False
        self.invert_assistant_lora = True

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading ZImage model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        self.print_and_status_update("Loading transformer")
        transformer_path = model_path
        transformer_subfolder = "transformer"
        if os.path.exists(transformer_path):
            transformer_subfolder = None
            transformer_path = os.path.join(transformer_path, "transformer")
            te_folder_path = os.path.join(model_path, "text_encoder")
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        # 1. Load Transformer
        transformer = ZImageTransformer2DModel.from_pretrained(
            transformer_path, 
            subfolder=transformer_subfolder, 
            torch_dtype=dtype,
            low_cpu_mem_usage=False, 
            ignore_mismatched_sizes=True
        )

        if self.model_config.assistant_lora_path is not None:
            self.load_training_adapter(transformer)

        # --- SURGICAL MODIFICATION: QUANTIZATION (QUANTO) ---
        should_quantize_transformer = self.model_config.quantize or (
            self.model_config.low_vram and not self.model_config.train_unet
        )
        
        if should_quantize_transformer:
            if self.model_config.qtype == "qfloat8":
                self.model_config.qtype = "float8"
                
            self.print_and_status_update(f"Surgical Plan: Quantizing Transformer (Quanto)")
            transformer.requires_grad_(False)
            
            # Monkey Patch BEFORE quantize
            transformer.requires_grad_ = types.MethodType(safe_requires_grad_, transformer)
            
            quantize_model(self, transformer)
            flush()
            
            # --- FIX: TROJAN HORSE PARAMETER ---
            transformer.dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=self.device_torch))
            transformer.dummy_param.requires_grad = True

            # Enable input grads (backup mechanism)
            if hasattr(transformer, "enable_input_require_grads"):
                transformer.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                if hasattr(transformer, "patch_embed"):
                    transformer.patch_embed.register_forward_hook(make_inputs_require_grad)
                elif hasattr(transformer, "pos_embed"):
                     transformer.pos_embed.register_forward_hook(make_inputs_require_grad)

        if (self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer.x_pad_token, transformer.cap_pad_token]
            )

        # --- SURGICAL FIX: RESIDENCY ---
        train_te = getattr(self.model_config, 'train_text_encoder', False)
        
        if self.model_config.low_vram and not train_te:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")
        else:
            self.print_and_status_update("Surgical Plan: Keeping Transformer on GPU for Gradient Flow")
            transformer.to(self.device_torch)

        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype
        )

        if (self.model_config.layer_offloading and self.model_config.layer_offloading_text_encoder_percent > 0):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        # --- SURGICAL MODIFICATION: TEXT ENCODER HANDLING ---
        if train_te:
            self.print_and_status_update("Surgical Plan: Text Encoder Training Active - Preserving BF16")
            text_encoder.to(self.device_torch, dtype=dtype)
            
            text_encoder.requires_grad_(False)
            if hasattr(text_encoder, "config"):
                text_encoder.config.use_cache = False
            
            self.print_and_status_update("Enabling Gradient Checkpointing for Text Encoder")
            text_encoder.gradient_checkpointing_enable()
            
            if hasattr(text_encoder, "enable_input_require_grads"):
                text_encoder.enable_input_require_grads()
            
            text_encoder.train()
        else:
            text_encoder.to(self.device_torch, dtype=dtype)
            if self.model_config.quantize_te:
                self.print_and_status_update("Quantizing Text Encoder")
                quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
                freeze(text_encoder)
                
        flush()

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype
        )

        self.noise_scheduler = ZImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {} # Fixed kwargs error

        pipe: ZImagePipeline = ZImagePipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
        )
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        text_encoder[0].to(self.device_torch)
        
        if not train_te:
            text_encoder[0].requires_grad_(False)
            text_encoder[0].eval()
            
        flush()

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.model = pipe.transformer
        
        # --- FIX: Alias UNet for BaseModel compatibility ---
        self.unet = self.model
        
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    # --- SURGICAL FIX: Custom Device State Handler ---
    def set_device_state(self, state):
        # Helper to get attributes safe for dict or object
        def get_state_attr(obj, name, default=None):
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        target_device = get_state_attr(state, 'device')
        
        if self.text_encoder is not None:
            if isinstance(self.text_encoder, list):
                for te in self.text_encoder:
                    te.to(target_device)
            else:
                self.text_encoder.to(target_device)
        
        if self.vae is not None:
             self.vae.to(target_device)
        
        if self.transformer is not None:
            self.transformer.to(target_device)
            should_train_unet = get_state_attr(state, 'train_unet', False)
            require_grads = get_state_attr(state, 'require_grads', False)
            
            if should_train_unet:
                self.transformer.train()
            else:
                self.transformer.eval()
                # Force train mode if using checkpointing for TE training
                if getattr(self.model_config, 'train_text_encoder', False):
                    self.transformer.train()
            
            # Apply grads SAFELY using monkey patched logic logic if needed, 
            # or manual check here
            target_grad = should_train_unet or require_grads
            for param in self.transformer.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad_(target_grad)
                else:
                    param.requires_grad_(False)


    def get_generation_pipeline(self):
        scheduler = ZImageModel.get_train_scheduler()
        pipeline: ZImagePipeline = ZImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )
        pipeline = pipeline.to(self.device_torch)
        return pipeline

    def generate_single_image(
        self,
        pipeline: ZImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)
        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            **extra,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)
        self.transformer.train() # Force train for checkpointing
        
        # --- SURGICAL FIX: Jumper Cable (External Checkpointing) ---
        
        # 1. Prepare Inputs (Must require grad)
        latent_model_input.requires_grad_(True)
        timestep_model_input = (1000 - timestep) / 1000
        
        encoder_hidden_states = text_embeddings.text_embeds
        if isinstance(encoder_hidden_states, list):
             encoder_hidden_states = encoder_hidden_states[0]
             text_embeddings.text_embeds = encoder_hidden_states
        
        # FIX: Ensure 3D (Batch, Seq, Dim)
        if torch.is_tensor(encoder_hidden_states):
            if encoder_hidden_states.ndim == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            encoder_hidden_states.requires_grad_(True)
            
        # 2. Define Wrapper (Handles unpacking for Z-Image)
        def _forward_wrapper(latents_batch, t_batch, encoder_hidden_batch):
            # Unpack Latents: (B, C, H, W) -> List[(C, 1, H, W)]
            latents_list = list(latents_batch.unsqueeze(2).unbind(0))
            
            # Unpack Embeds: (B, Seq, Dim) -> List[(Seq, Dim)]
            # Fixes "1 and 2" dimensions error by strictly providing 2D tensors
            encoder_list = list(encoder_hidden_batch.unbind(0))
            
            # Run Transformer (Black Box)
            output = self.transformer(latents_list, t_batch, encoder_list)
            
            # Repack: List[Tensor] -> Tensor
            return torch.stack([t.float() for t in output[0]], dim=0)

        # 3. Execute via Jumper Cable
        noise_pred = torch.utils.checkpoint.checkpoint(
            _forward_wrapper,
            latent_model_input,
            timestep_model_input,
            encoder_hidden_states,
            use_reentrant=False 
        )

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred
        
        # --- SURGICAL FIX: TROJAN HORSE CONNECTION ---
        if hasattr(self.transformer, "dummy_param"):
            if self.transformer.dummy_param.device != noise_pred.device:
                 self.transformer.dummy_param.data = self.transformer.dummy_param.data.to(noise_pred.device)
            loss_proxy = self.transformer.dummy_param.sum() * 0
            noise_pred = noise_pred + loss_proxy

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        # SURGICAL FIX: Manual Pipeline Bypass & Direct Input Injection
        train_te = getattr(self.model_config, 'train_text_encoder', False)
        
        if not train_te:
            if self.pipeline.text_encoder.device != self.device_torch:
                self.pipeline.text_encoder.to(self.device_torch)
            prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt,
                do_classifier_free_guidance=False,
                device=self.device_torch,
            )
            return PromptEmbeds([prompt_embeds, None])
        
        tokenizer = self.tokenizer[0] if isinstance(self.tokenizer, list) else self.tokenizer
        text_encoder = self.text_encoder[0] if isinstance(self.text_encoder, list) else self.text_encoder
        
        text_encoder.to(self.device_torch)
        
        if isinstance(prompt, str):
            prompt = [prompt]
            
        max_len = getattr(tokenizer, 'model_max_length', 512)
        if max_len > 1024: max_len = 512 
        
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.device_torch)
        
        # Direct Injection Logic
        with torch.set_grad_enabled(True):
            input_embed_layer = text_encoder.get_input_embeddings()
            inputs_embeds = input_embed_layer(text_inputs.input_ids)
            inputs_embeds.requires_grad_(True)
            
            outputs = text_encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=text_inputs.attention_mask,
                output_hidden_states=True
            )
            
            if hasattr(outputs, "hidden_states"):
                prompt_embeds = outputs.hidden_states[-1]
            else:
                prompt_embeds = outputs[0]
                
            # FORCE 3D SHAPE (Batch, Seq, Dim)
            if prompt_embeds.ndim == 2:
                prompt_embeds = prompt_embeds.unsqueeze(0)

        return PromptEmbeds([prompt_embeds, None])

    def get_model_has_grad(self):
        if self.model is None:
            return False
        return any(p.requires_grad for p in self.model.parameters())

    def get_te_has_grad(self):
        if self.text_encoder is None:
            return False
        te0 = self.text_encoder[0] if isinstance(self.text_encoder, list) else self.text_encoder
        return any(p.requires_grad for p in te0.parameters())

    def save_model(self, output_path, meta, save_dtype):
        transformer: ZImageTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        if self.get_te_has_grad():
            te0 = self.text_encoder[0] if isinstance(self.text_encoder, list) else self.text_encoder
            te0 = unwrap_model(te0)
            te0.save_pretrained(
                save_directory=os.path.join(output_path, "text_encoder"),
                safe_serialization=True,
            )
            tok0 = self.tokenizer[0] if isinstance(self.tokenizer, list) else self.tokenizer
            tok0.save_pretrained(os.path.join(output_path, "tokenizer"))

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "zimage"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd