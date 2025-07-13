import gc
from typing import Optional
from PIL import Image
import numpy as np
import math

import torch
import torch.nn as nn
import transformer_lens.HookedTransformer as HookedTransformer
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
from transformers import BitsAndBytesConfig


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


# Copied from transformers.models.llava_next.modeling_llava_next.image_size_to_num_patches
def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
        """
        Calculate the number of patches after the preprocessing for images of any resolution.

        Args:
            image_size (`torch.LongTensor` or `np.ndarray` or `Tuple[int, int]`):
                The size of the input image in the format (height, width). ?
            grid_pinpoints (`List`):
                A list containing possible resolutions. Each item in the list should be a tuple or list
                of the form `(height, width)`.
            patch_size (`int`):
                The size of each image patch.

        Returns:
            int: the number of patches
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints should be a list of tuples or lists")

        # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
        if not isinstance(image_size, (list, tuple)):
            if not isinstance(image_size, (torch.Tensor, np.ndarray)):
                raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
            image_size = image_size.tolist()

        best_resolution = select_best_resolution(image_size, grid_pinpoints)
        height, width = best_resolution
        num_patches = 0
        # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                num_patches += 1
        # add the base patch
        num_patches += 1
        return num_patches

# Copied from transformers.models.llava_next.modeling_llava_next.get_anyres_image_grid_shape
def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


# Copied from transformers.models.llava_next.modeling_llava_next.unpad_image
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class HookedLlavaOneVision():

    def __init__(self, cache_dir, device='cuda', num_devices=1, load_in_4bit=False, model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"):
        
        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
        self.dtype = torch.float32

        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=self.dtype, 
            low_cpu_mem_usage=True, 
            cache_dir = cache_dir,
            device_map = 'cuda' if load_in_4bit else 'cpu',
            attn_implementation="eager",
            quantization_config=quantization_config
        ).eval()

        if load_in_4bit:
            model.language_model.config.quantization_config = dict(load_in_4bit=True, quant_method='bitsandbytes')

        self.processor = LlavaOnevisionProcessor.from_pretrained(model_id, cache_dir = cache_dir)
        self.config = model.config

        self.vision_tower = model.vision_tower.eval()
        self.multi_modal_projector = model.multi_modal_projector.eval()
        self.image_newline = model.image_newline

        self.input_embeddings = model.get_input_embeddings()
        self.hooked_language_model = HookedTransformer.from_pretrained(
            'Qwen/Qwen2-7B',
            hf_model=model.language_model,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            dtype=self.dtype
        )
        self.hooked_language_model.set_tokenizer(self.processor.tokenizer, 
                                                 default_padding_side=self.processor.tokenizer.padding_side)
        self.hooked_language_model.eval()

        del model
        gc.collect()


    def pack_image_features(self, image_features, image_sizes, image_newline=None, vision_aspect_ratio="anyres_max_9"):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
            vision_aspect_ratio (`str`, *optional*, "anyres_max_9"):
                Aspect ratio used when processong image features. The default value is "anyres_max_9".
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                max_num_patches = int(vision_aspect_ratio.strip("anyres_max_"))
                channels, curr_height, curr_width = image_feature.shape
                ratio = math.sqrt(curr_height * curr_width / (max_num_patches * height**2))
                if ratio > 1.1:
                    image_feature = image_feature[None]
                    image_feature = nn.functional.interpolate(
                        image_feature, [int(curr_height // ratio), int(curr_width // ratio)], mode="bilinear"
                    )[0]
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens



    @torch.no_grad
    def pre_forward(self, input_ids, image_sizes, pixel_values):
        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy
        vision_aspect_ratio = self.config.vision_aspect_ratio
        inputs_embeds = self.input_embeddings(input_ids)

        image_num_patches = [
                image_size_to_num_patches(
                    image_size=imsize,
                    grid_pinpoints=self.config.image_grid_pinpoints,
                    patch_size=self.config.vision_config.image_size,
                )
                for imsize in image_sizes
            ]
        
        # unpad extra patches and concatenate them
        if pixel_values.dim() == 5:
            _pixel_values_list = [
                pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            # [batch_size*frames*num_patches, num_channels, height, width] where frames=1 for images
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")
        
        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)

        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, feature_lens = self.pack_image_features(
            image_features,
            image_sizes,
            image_newline=self.image_newline,
            vision_aspect_ratio=vision_aspect_ratio,
        )

        special_image_mask = (
            (input_ids == self.config.image_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds
    

    @torch.no_grad
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_sizes: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None
    ):

        inputs_embeds = self.pre_forward(input_ids=input_ids, image_sizes=image_sizes, pixel_values=pixel_values)
        outputs = self.hooked_language_model(
                inputs_embeds, start_at_layer=0, shortformer_pos_embed=None, attention_mask=attention_mask
        )

        return outputs
    

    @torch.no_grad
    def run_with_cache(
        self,
        input_ids: torch.LongTensor = None,
        image_sizes: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        names_filter = None,
        inputs_embeds = None
    ):
        if inputs_embeds == None:
            inputs_embeds = self.pre_forward(input_ids=input_ids, image_sizes=image_sizes, pixel_values=pixel_values)
        logits, cache = self.hooked_language_model.run_with_cache(
            inputs_embeds, names_filter=names_filter, start_at_layer=0, shortformer_pos_embed=None, attention_mask=attention_mask
        )

        return logits, cache
    
    @torch.no_grad  
    def run_with_hooks(self, input_ids, pixel_values, image_sizes, attention_mask, fwd_hooks = []):
        
        inputs_embeds = self.pre_forward(input_ids=input_ids, image_sizes=image_sizes, pixel_values=pixel_values)
        logits = self.hooked_language_model.run_with_hooks(
            inputs_embeds, start_at_layer=0, shortformer_pos_embed=None, fwd_hooks=fwd_hooks, attention_mask=attention_mask
        )

        return logits
    

    @torch.no_grad
    def run_with_cache_and_hooks(self, input_ids, pixel_values, image_sizes, attention_mask, fwd_hooks = [], names_filter=None):

        inputs_embeds = self.pre_forward(input_ids=input_ids, image_sizes=image_sizes, pixel_values=pixel_values)

        with self.hooked_language_model.hooks(fwd_hooks=fwd_hooks):
            logits, cache = self.hooked_language_model.run_with_cache(
                inputs_embeds, names_filter=names_filter, start_at_layer=0, shortformer_pos_embed=None, attention_mask=attention_mask
            )
            
        return logits, cache
    
    
    def to(self, device):
        self.vision_tower.to(device)
        self.multi_modal_projector.to(device)
        self.input_embeddings.to(device)
        self.image_newline.to(device)



def fetch_llava_ov_model(cache_dir='/hf/hub', device='cpu', num_devices=1, load_in_4bit=False, ov72b=False):
    
    model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf" if ov72b else "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    model = HookedLlavaOneVision(cache_dir=cache_dir,device=device,num_devices=num_devices,load_in_4bit=load_in_4bit, model_id=model_id)
    return model
