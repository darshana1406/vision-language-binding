from utils.helper import *
from utils.data import *
from utils.details import *
from utils.interventions import *
from utils.metrics import *

from hooked_llava_ov import fetch_llava_ov_model

from PIL import Image
from functools import partial
import torch
from jaxtyping import Float, Int, Bool
import gc
from tqdm import tqdm
from collections import defaultdict
from einops import repeat
import einops
import numpy as np
import argparse

from transformer_lens.utils import get_act_name
import transformer_lens.HookedTransformer as HookedTransformer


def random_vector_like(tens):
    norms = torch.linalg.vector_norm(tens, dim=-1, keepdim=True)
    noise = torch.normal(torch.zeros_like(tens), 1)
    noise /= torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
    return noise * norms

def mean_ab(*, model, vocab, num_samples, device, batch_size, num_image_patches, num_layers, intervention, mean_shape=None, mean_item=None, mean_color=None):

    start = 0
    num_samples = start+num_samples
    scores = torch.zeros((2,2))
    log_probs = torch.zeros(2, num_samples, 2)


    for batch_idx, start_idx in enumerate(tqdm(range(start, num_samples, batch_size))):

        # start_idx = batch_idx * batch_size
        end_idx = min(start_idx+batch_size, num_samples)
        bs= end_idx - start_idx


        ctx1_list, _ = zip(*vocab.get_sample_dicts(list(range(start_idx,end_idx))))
        ctx1_text_list, ctx1_indices_list, _, _ = zip(*vocab.get_samples(list(range(start_idx,end_idx)),0))

        ctx1_aligned_indices = [align_token_indices(
            model.processor.tokenizer,
            ctx1_text_list[i],
            ctx1_indices_list[i],
            num_image_patches=num_image_patches
        )[1] for i in range(bs)]

        ctx1_img_list = [Image.open(ctx1['image_path']).convert('RGB') for ctx1 in ctx1_list]
        ctx1_inputs = model.processor(images=ctx1_img_list, text=ctx1_text_list, return_tensors="pt").to(device, dtype=model.dtype)

        _, ctx1_cache = model.run_with_cache(**ctx1_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))

        for ele in intervention:
            if ele=='shape' and not(mean_shape == None):
                shape1_patches, shape2_patches = vocab.image_patches['shape1'], vocab.image_patches['shape2']
                for i,img_pos in enumerate(shape2_patches):
                    ctx1_cache = update_cache(ctx1_cache, img_pos, -mean_shape[i], num_layers, 'resid_pre')
                for i,img_pos in enumerate(shape1_patches):
                    ctx1_cache = update_cache(ctx1_cache, img_pos, mean_shape[i], num_layers, 'resid_pre')

            
            if ele=='item' and not(mean_item == None):
                pos1, pos2 = ctx1_aligned_indices[0][f'item1'], ctx1_aligned_indices[0][f'item2']
                padding = 1
                pos1, pos2 = Substring(pos1[0]-padding, pos1[1]), Substring(pos2[0]-padding, pos2[1])

                ctx1_cache = update_cache(ctx1_cache, pos2, -mean_item, num_layers, 'resid_pre')
                ctx1_cache = update_cache(ctx1_cache, pos1, mean_item, num_layers, 'resid_pre')

            if ele=='color' and not(mean_color == None):
                pos1, pos2 = ctx1_aligned_indices[0][f'color1'], ctx1_aligned_indices[0][f'color2']
                padding = 1
                pos1, pos2 = Substring(pos1[0], pos1[1]+padding), Substring(pos2[0], pos2[1]+padding)

                ctx1_cache = update_cache(ctx1_cache, pos2, -mean_color, num_layers, 'resid_pre')
                ctx1_cache = update_cache(ctx1_cache, pos1, mean_color, num_layers, 'resid_pre')


        target_dependent_mask = torch.zeros(
            (bs, 28, ctx1_inputs.input_ids.shape[1]), dtype=bool
        )
        dependent_start = ctx1_aligned_indices[0]['context'][1]
        target_dependent_mask[:, :, dependent_start:] = True
        source_mask = ~target_dependent_mask



        fwd_hooks = []
        for layer in range(28):
            fwd_hooks.append((
                get_act_name('resid_pre', layer),
                partial(patch_masked_residue,mask=source_mask[:,layer,:],source_cache=ctx1_cache)
            ))
            # fwd_hooks.append((
            #     get_act_name('rot_k', layer),
            #     partial(patch_rotary_k, 
            #             pos_deltas=pos_deltas,
            #             rotate_function=partial(rotary_deltas, attn=model.hooked_language_model.blocks[layer].attn)
            #     )
            # ))
           
        for shape_id in [0,1]:
                ctx1_qn,_,_,_ = zip(*vocab.get_samples(list(range(start_idx,end_idx)),shape_id))
                inputs = model.processor(images=ctx1_img_list, text=ctx1_qn, return_tensors='pt').to(device)
                logits = model.run_with_hooks(**inputs, fwd_hooks=fwd_hooks)

                for i, ctx1 in enumerate(ctx1_list):
                    idx = start_idx+i
                    item_logits, score = evaluate(logits[i], ctx1, model.processor.tokenizer)
                    log_probs[shape_id,idx] = item_logits
                    scores[shape_id] += score

        print(idx,scores)


    # med_acc = median_calib_acc(log_probs)

    log_probs_summary = log_probs.mean(dim=-2)

    print('__________________Accuracy_________________')
    print(scores)

    # print('__________________Calib Accuracy_________________')
    # print(med_acc)

    print('__________________Mean Log Probs_________________')
    print(log_probs_summary)

    return dict(accuracy=scores, log_probs=log_probs)


def binding_change(*, model, vocab, num_samples, device, batch_size, num_image_patches, num_layers, interventions, baseline=False):
    path_t = 'outputs/mean_ab_{}.pth'
    mean_shape = torch.load('outputs/binding_means/shape.pt')
    mean_item = torch.load('outputs/binding_means/item.pt')
    mean_color = torch.load('outputs/binding_means/color.pt')

    if baseline:
        path_t = 'outputs/random_mean_ab_{}.pth'
        mean_shape = random_vector_like(mean_shape)
        mean_item = random_vector_like(mean_item)
        mean_color = random_vector_like(mean_color)

    for inter in interventions:
        result = mean_ab(model=model, vocab=vocab, num_samples=num_samples, device=device, batch_size=1, intervention=inter,
                          num_image_patches=(27*27)+(28*27), num_layers=num_layers, mean_shape=mean_shape, mean_item=mean_item, mean_color=mean_color)
        path = path_t.format('_'.join(inter)) 
        torch.save(result,path)       


def run_mean_ab(args):

    dataset_name = 'shapes'
    im_flip = False

    vocab = ShapesItems(csv_path='dataset/factorizability.csv', 
                        img_path='', c_mapping=False, s_mapping=True, flip=False)
    
    llava_ov_deets = get_llava_ov_details(w=3,  im_flip=im_flip)
    vocab.set_details(**llava_ov_deets, flip=False)

    num_samples = 5#len(vocab.rows)

    device = torch.device('cuda')
    model = fetch_llava_ov_model(cache_dir=args.cache_dir, device='cuda',num_devices=1, load_in_4bit=False)

    model.to(device=device)
    num_layers = model.hooked_language_model.cfg.n_layers #28

    interventions = [['item'], ['shape'], ['shape','item']]

    binding_change(model=model, vocab=vocab, num_samples=num_samples, device=device, batch_size=1, num_image_patches=(27*27)+(28*27), num_layers=num_layers, interventions=interventions, baseline=args.random_means)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, help='HF cache dir path')
    parser.add_argument("--random_means", action='store_true')

    args = parser.parse_args()

    run_mean_ab(args)
