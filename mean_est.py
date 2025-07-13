from utils.data import *
from utils.helper import *
from utils.details import *
from hooked_llava_ov import fetch_llava_ov_model

from PIL import Image
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


def extract_mean_item(*, model, vocab, num_samples, device, batch_size, num_image_patches, num_layers):
    start = 0
    num_samples = start+num_samples
    scores = torch.zeros((2,2))
    log_probs = torch.zeros(2, num_samples, 2)

    name = 'item'
    b_list = {name:[]}
    cnt = {name:0}

    for batch_idx, start_idx in enumerate(tqdm(range(start, num_samples, batch_size))):

        # start_idx = batch_idx * batch_size
        end_idx = min(start_idx+batch_size, num_samples)
        bs= end_idx - start_idx

        _, ctx2_list = zip(*vocab.get_sample_dicts(list(range(start_idx,end_idx))))
        _, _, ctx2_text_list, ctx2_indices_list, = zip(*vocab.get_samples(list(range(start_idx,end_idx)),3, flip_item=False))
        _, _, ctx2_flip_text_list, _, = zip(*vocab.get_samples(list(range(start_idx,end_idx)),3, flip_item=True))

        ctx2_aligned_indices = [align_token_indices(
            model.processor.tokenizer,
            ctx2_text_list[i],
            ctx2_indices_list[i],
            num_image_patches=num_image_patches
        )[1] for i in range(bs)]

        ctx2_img_list = [Image.open(ctx2['image_path']).convert('RGB') for ctx2 in ctx2_list]
        ctx2_inputs = model.processor(images=ctx2_img_list, text=ctx2_text_list, return_tensors="pt").to(device, dtype=model.dtype)

        ctx2_flip_inputs = model.processor(images=ctx2_img_list, text=ctx2_flip_text_list, return_tensors="pt").to(device, dtype=model.dtype)

        _, ctx2_cache = model.run_with_cache(**ctx2_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))
        _, ctx2_flip_cache = model.run_with_cache(**ctx2_flip_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))

        ctx2_cache = torch.stack(list(ctx2_cache.values())) # l,b,w,dim
        ctx2_flip_cache = torch.stack(list(ctx2_flip_cache.values())) # l,b,w,dim

        pos1, pos2 = ctx2_aligned_indices[0][f'{name}1'], ctx2_aligned_indices[0][f'{name}2']
        pos1, pos2 = Substring(pos1[0]-1, pos1[1]), Substring(pos2[0]-1, pos2[1])
        b1 = (ctx2_cache[:,:,pos2.to_slice()] - ctx2_flip_cache[:,:,pos1.to_slice()]).mean(1)
        b2 = (ctx2_flip_cache[:,:,pos2.to_slice()] - ctx2_cache[:,:,pos1.to_slice()]).mean(1)

        if len(b_list[name]) == 0:
            b_list[name].append(b1)
            b_list[name][0] += b2
        else:
            b_list[name][0] += b1
            b_list[name][0] += b2
        cnt[name] += 2


    print(cnt)
    for k in b_list:
        # tensor = torch.stack(b_list[k]).mean(0)
        b_list[k][0] /= cnt[k]
    
    mean = b_list[name][0]#.mean(dim=(0,2))
    print(mean.shape)
    return mean



def extract_mean_shape(*, model, vocab, num_samples, device, batch_size, num_image_patches, num_layers):
    start = 0
    num_samples = start+num_samples
    scores = torch.zeros((2,2))
    log_probs = torch.zeros(2, num_samples, 2)

    b_list = {'shape':[]}

    for batch_idx, start_idx in enumerate(tqdm(range(start, num_samples, batch_size))):

        # start_idx = batch_idx * batch_size
        end_idx = min(start_idx+batch_size, num_samples)
        bs= end_idx - start_idx


        ctx1_list, ctx2_list = zip(*vocab.get_sample_dicts(list(range(start_idx,end_idx))))
        ctx1_text_list, ctx1_indices_list, ctx2_text_list, ctx2_indices_list, = zip(*vocab.get_samples(list(range(start_idx,end_idx)),3))

        ctx2_aligned_indices = [align_token_indices(
            model.processor.tokenizer,
            ctx2_text_list[i],
            ctx2_indices_list[i],
            num_image_patches=num_image_patches
        )[1] for i in range(bs)]
        
        ctx2_img_list = [Image.open(ctx2['image_path']).convert('RGB') for ctx2 in ctx2_list]
        ctx2_inputs = model.processor(images=ctx2_img_list, text=ctx2_text_list, return_tensors="pt").to(device, dtype=model.dtype)

        ctx1_img_list = [Image.open(ctx1['image_path']).convert('RGB') for ctx1 in ctx1_list]
        ctx1_inputs = model.processor(images=ctx1_img_list, text=ctx1_text_list, return_tensors="pt").to(device, dtype=model.dtype)

        
        pos1_l, pos2_l = vocab.image_patches['shape1'], vocab.image_patches['shape2']

        _, ctx1_cache = model.run_with_cache(**ctx1_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))
        _, ctx2_cache = model.run_with_cache(**ctx2_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))
        ctx2_cache = torch.stack(list(ctx2_cache.values())) # l,b,w,dim
        ctx1_cache = torch.stack(list(ctx1_cache.values())) # l,b,w,dim


        b1,b2 = [],[]
        for pos1, pos2 in zip(pos1_l, pos2_l):
            _b1 = (ctx2_cache[:,:,pos2.to_slice()] - ctx1_cache[:,:,pos1.to_slice()]).mean(1) # mean removes batch dim
            _b2 = (ctx1_cache[:,:,pos2.to_slice()] - ctx2_cache[:,:,pos1.to_slice()]).mean(1)
            b1.append(_b1)
            b2.append(_b2)

        name = 'shape'
        if len(b_list[name]) == 0:
            b_list[name].append(torch.stack(b1))
        else:
            b_list[name][0] += torch.stack(b1)
        b_list[name][0] += torch.stack(b2)

    # pos,l, pathches, dim
    # torch.Size([22, 28, 11, 3584])
    b_list[name][0] /= (num_samples*2)
    mean = b_list[name][0]#.mean(dim=(0,2))

    return mean



def extract_mean_color(*, model, vocab, num_samples, device, batch_size, num_image_patches, num_layers):
    start = 0
    num_samples = start+num_samples
    scores = torch.zeros((2,2))
    log_probs = torch.zeros(2, num_samples, 2)

    name = 'color'
    b_list = {name:[]}
    cnt = {name:0}

    for batch_idx, start_idx in enumerate(tqdm(range(start, num_samples, batch_size))):

        # start_idx = batch_idx * batch_size
        end_idx = min(start_idx+batch_size, num_samples)
        bs= end_idx - start_idx

        _, ctx2_list = zip(*vocab.get_sample_dicts(list(range(start_idx,end_idx))))
        _, _, ctx2_text_list, ctx2_indices_list, = zip(*vocab.get_samples(list(range(start_idx,end_idx)),0))
        _, _, ctx1_text_list, ctx1_indices_list = zip(*vocab.get_samples(list(range(start_idx,end_idx)),0, order_flip=True))

        ctx2_aligned_indices = [align_token_indices(
            model.processor.tokenizer,
            ctx2_text_list[i],
            ctx2_indices_list[i],
            num_image_patches=num_image_patches
        )[1] for i in range(bs)]
        
        ctx2_img_list = [Image.open(ctx2['image_path']).convert('RGB') for ctx2 in ctx2_list]
        ctx2_inputs = model.processor(images=ctx2_img_list, text=ctx2_text_list, return_tensors="pt").to(device, dtype=model.dtype)

        ctx1_img_list = [Image.open(ctx1['image_path']).convert('RGB') for ctx1 in ctx2_list]
        ctx1_inputs = model.processor(images=ctx1_img_list, text=ctx1_text_list, return_tensors="pt").to(device, dtype=model.dtype)

        _, ctx1_cache = model.run_with_cache(**ctx1_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))
        _, ctx2_cache = model.run_with_cache(**ctx2_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))
        ctx2_cache = torch.stack(list(ctx2_cache.values())) # l,b,w,dim
        ctx1_cache = torch.stack(list(ctx1_cache.values())) # l,b,w,dim

        pos1, pos2 = ctx2_aligned_indices[0][f'{name}1'], ctx2_aligned_indices[0][f'{name}2']
        padding = 1
        pos1, pos2 = Substring(pos1[0], pos1[1]+padding), Substring(pos2[0], pos2[1]+padding)
        b1 = (ctx2_cache[:,:,pos2.to_slice()] - ctx1_cache[:,:,pos1.to_slice()]).mean(1)
        b2 = (ctx1_cache[:,:,pos2.to_slice()] - ctx2_cache[:,:,pos1.to_slice()]).mean(1)

        if len(b_list[name]) == 0:
            b_list[name].append(b1)
            b_list[name][0] += b2
        else:
            b_list[name][0] += b1
            b_list[name][0] += b2
        cnt[name] += 2


    print(cnt)
    for k in b_list:
        # tensor = torch.stack(b_list[k]).mean(0)
        b_list[k][0] /= cnt[k]
    
    mean = b_list[name][0]#.mean(dim=(0,2))
    print(mean.shape)
    return mean


def mean_est(*, model, device, batch_size, num_image_patches, num_layers):

    vocab = ShapesItems(csv_path='/home/darshana/research/vlm-bind/dataset/shape_mean.csv', img_path='', c_mapping=False, s_mapping=True, flip=False)
    llava_ov_deets = get_llava_ov_details_mean_est(w=3)
    vocab.set_details(**llava_ov_deets, flip=False)

    shape_means = extract_mean_shape(model=model, vocab=vocab, num_samples=len(vocab.rows), device=device, 
                               batch_size=batch_size, num_image_patches=num_image_patches, num_layers=num_layers)
    torch.save(shape_means, 'outputs/binding_means/shape.pt')

    vocab = ShapesItems(csv_path='/home/darshana/research/vlm-bind/dataset/color_mean.csv', img_path='', c_mapping=False, s_mapping=True, flip=False)
    llava_ov_deets = get_llava_ov_details_mean_est(w=3)
    vocab.set_details(**llava_ov_deets, flip=False)

    color_means = extract_mean_color(model=model, vocab=vocab, num_samples=len(vocab.rows), device=device, 
                               batch_size=batch_size, num_image_patches=num_image_patches, num_layers=num_layers)
    torch.save(color_means, 'outputs/binding_means/color.pt')
    

    vocab = ShapesItems(csv_path='/home/darshana/research/vlm-bind/dataset/item_mean.csv', img_path='', c_mapping=False, s_mapping=True, flip=False)
    llava_ov_deets = get_llava_ov_details_mean_est(w=3)
    vocab.set_details(**llava_ov_deets, flip=False)

    item_means = extract_mean_item(model=model, vocab=vocab, num_samples=len(vocab.rows), device=device, 
                               batch_size=batch_size, num_image_patches=num_image_patches, num_layers=num_layers)
    torch.save(item_means, 'outputs/binding_means/item.pt')



def run_mean_est_llava_ov(args):

    dataset_name = 'shapes'


    device = torch.device('cuda')
    # device = torch.device('cpu')
    model = fetch_llava_ov_model(cache_dir=args.cache_dir, device='cuda',num_devices=1)

    model.to(device=device)
    num_layers = model.hooked_language_model.cfg.n_layers #28

    mean_est(model=model, device=device, batch_size=1, num_image_patches=(27*27)+(28*27), num_layers=num_layers)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, help='HF cache dir path')

    args = parser.parse_args()

    run_mean_est_llava_ov(args)