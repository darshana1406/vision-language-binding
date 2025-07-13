from utils.data import *
from utils.helper import *
from utils.details import *
from utils.interventions import *
from utils.metrics import *
from hooked_llava_ov import fetch_llava_ov_model

from PIL import Image
from functools import partial
import torch
from collections import defaultdict
from tqdm import tqdm
import argparse

from transformer_lens.utils import get_act_name

def intervene(*, model, vocab, num_samples, num_layers, num_image_patches, device, batch_size, intervene_name):
    position_scores = defaultdict(lambda: torch.zeros((2,2)))
    position_log_probs = defaultdict(lambda: torch.zeros(2, num_samples, 2))
    log_probs_summary = {}

    for batch_idx, start_idx in enumerate(tqdm(range(0, num_samples, batch_size))):

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
        ctx1_inputs = model.processor(images=ctx1_img_list, text=ctx1_text_list, return_tensors="pt").to(device)

        _, ctx1_cache = model.run_with_cache(**ctx1_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))

        target_dependent_mask = torch.zeros(
            (bs, num_layers, ctx1_inputs.input_ids.shape[1]), dtype=bool
        )
        dependent_start = ctx1_aligned_indices[0]['context'][1]
        target_dependent_mask[:, :, dependent_start:] = True
        source_mask = ~target_dependent_mask

        if intervene_name == 'item':
            pos1, pos2 = ctx1_aligned_indices[0]['item1'], ctx1_aligned_indices[0]['item2']
            pos1, pos2 = Substring(pos1[0]-1, pos1[1]), Substring(pos2[0]-1, pos2[1])
            pos1_l, pos2_l = [pos1], [pos2]
            delta_list = [-3, 0, 3, 6, 7, 9]
        elif intervene_name == 'color':
            pos1, pos2 = ctx1_aligned_indices[0]['color1'], ctx1_aligned_indices[0]['color2']
            pos1, pos2 = Substring(pos1[0], pos1[1]+1), Substring(pos2[0], pos2[1]+1)
            pos1_l, pos2_l = [pos1], [pos2]
            delta_list = [-3, 0, 3, 6, 7, 9]
        elif intervene_name == 'shape':
            pos1_l, pos2_l = vocab.image_patches['shape1'], vocab.image_patches['shape2']
            delta_list = [-30, 0, 30, 60, 90, 95, 98, 100, 120]

        for delta in delta_list:

            pos_deltas = torch.zeros((bs, ctx1_inputs.input_ids.shape[1]), dtype=int)
            for pos1, pos2 in zip(pos1_l, pos2_l):
                pos_deltas[:,pos1.to_slice()] = delta
                pos_deltas[:,pos2.to_slice()] = -delta

            fwd_hooks = []
            for layer in range(num_layers):
                fwd_hooks.append((
                    get_act_name('resid_pre', layer),
                    partial(patch_masked_residue,mask=source_mask[:,layer,:],source_cache=ctx1_cache)
                ))
                fwd_hooks.append((
                    get_act_name('rot_k', layer),
                    partial(patch_rotary_k, 
                            pos_deltas=pos_deltas,
                            rotate_function=partial(rotary_deltas, attn=model.hooked_language_model.blocks[layer].attn)
                    )
                ))

            for shape_id in range(2):
                ctx1_qn,_,ctx2_qn,_ = zip(*vocab.get_samples(list(range(start_idx,end_idx)),shape_id))
                inputs = model.processor(images=ctx1_img_list, text=ctx1_qn, return_tensors='pt').to(device)
                logits = model.run_with_hooks(**inputs, fwd_hooks=fwd_hooks)

                for i, ctx1 in enumerate(ctx1_list):
                    idx = start_idx+i
                    item_logits, score = evaluate(logits[i], ctx1, model.processor.tokenizer)
                    position_log_probs[delta][shape_id,idx] = item_logits
                    position_scores[delta][shape_id] += score

        print(idx,'____________Scores________________')
        print(*position_scores.keys())
        print(*(position_scores.values()),sep='\n\n')
        # print(idx,'____________Log_Probs______________')
        # print(*position_log_probs.keys())
        # print(*(position_log_probs.values()), sep='\n\n')

    for key in position_log_probs:
        log_probs = position_log_probs[key].mean(dim=-2)
        log_probs_summary[key] = log_probs

    
    print('____________Scores________________')
    print(*position_scores.keys())
    print(*(position_scores.values()),sep='\n\n')

    print('____________Log_Probs______________')
    print(*log_probs_summary.keys())
    print(*(log_probs_summary.values()), sep='\n\n')

    return dict(accuracy=dict(position_scores), log_probs=dict(position_log_probs))



def run_pos_ind_llava_ov(args):

    dataset_name = 'shapes'
    im_flip = False

    save_path_temp = 'outputs/pos_ind_{}.pth'

    vocab = ShapesItems(csv_path='dataset/factorizability.csv', 
                        img_path='', c_mapping=False, s_mapping=True, flip=False)
    llava_ov_deets = get_llava_ov_details(w=3,  im_flip=im_flip)
    vocab.set_details(**llava_ov_deets, flip=False)


    num_samples = len(vocab.rows)

    device = torch.device('cuda')
    model = fetch_llava_ov_model(cache_dir=args.cache_dir, device='cuda',num_devices=1)
    model.to(device=device)
    num_layers = model.hooked_language_model.cfg.n_layers

    intervene_list = ['shape','item','color']
    for name in intervene_list:
        result = intervene(model=model, vocab=vocab, num_samples=num_samples, num_image_patches=(27*27)+(28*27),
                           num_layers=num_layers, device=device, batch_size=1, intervene_name=name)
        torch.save(result, save_path_temp.format(name))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, help='HF cache dir path')

    args = parser.parse_args()

    run_pos_ind_llava_ov(args)
