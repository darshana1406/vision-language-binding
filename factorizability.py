from utils.data import *
from utils.helper import *
from utils.details import *
from hooked_llava_ov import fetch_llava_ov_model
from utils.interventions import *
from utils.metrics import *

from PIL import Image
from functools import partial
import torch
from jaxtyping import Float, Int, Bool
import gc
from tqdm import tqdm
import argparse

from transformer_lens.utils import get_act_name


def intervene(*, model, vocab, num_samples, device, num_layers, intervene_list, bs, num_image_patches):
    counts = torch.zeros((4,4))
    log_probs_matrix = torch.zeros((4,num_samples, 4))

    for batch_idx, start_idx in enumerate(tqdm(range(0, num_samples, bs))):

        # start_idx = batch_idx * batch_size
        end_idx = min(start_idx+bs, num_samples)
        batch_size = end_idx - start_idx

        ctx1_list, ctx2_list = zip(*vocab.get_sample_dicts(list(range(start_idx,end_idx))))
        ctx1_text_list, ctx1_indices_list, ctx2_text_list, ctx2_indices_list = zip(*vocab.get_samples(list(range(start_idx,end_idx)),0))

        ctx1_aligned_indices = [align_token_indices(
            model.processor.tokenizer,
            ctx1_text_list[i],
            ctx1_indices_list[i],
            num_image_patches=num_image_patches
        )[1] for i in range(batch_size)]
        ctx2_aligned_indices = [align_token_indices(
            model.processor.tokenizer,
            ctx2_text_list[i],
            ctx2_indices_list[i],
            num_image_patches=num_image_patches
        )[1] for i in range(batch_size)]
        
        ctx1_img_list = [Image.open(ctx1['image_path']).convert('RGB') for ctx1 in ctx1_list]
        ctx2_img_list = [Image.open(ctx2['image_path']).convert('RGB') for ctx2 in ctx2_list]

        ctx1_inputs = model.processor(images=ctx1_img_list, text=ctx1_text_list, return_tensors="pt").to(device)
        ctx2_inputs = model.processor(images=ctx2_img_list, text=ctx2_text_list, return_tensors="pt").to(device)

        _, ctx2_cache = model.run_with_cache(**ctx2_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))
        _, ctx1_cache = model.run_with_cache(**ctx1_inputs, names_filter=lambda x: any(y in x for y in ['resid_pre']))

        alter_cache_i1 = ctx1_cache
        for intervene in intervene_list:
            if intervene.startswith('item'):
                pos1, pos2 = ctx1_aligned_indices[0][intervene], ctx2_aligned_indices[0][intervene]
                pos1, pos2 = Substring(pos1[0]-1, pos1[1]), Substring(pos2[0]-1, pos2[1])
                alter_cache_i1 = replace_cache(alter_cache_i1, ctx2_cache, pos1, pos2, num_layers, 'resid_pre')
            elif intervene.startswith('color'):
                pos1, pos2 = ctx1_aligned_indices[0][intervene], ctx2_aligned_indices[0][intervene]
                pos1, pos2 = Substring(pos1[0], pos1[1]+1), Substring(pos2[0], pos2[1]+1)
                alter_cache_i1 = replace_cache(alter_cache_i1, ctx2_cache, pos1, pos2, num_layers, 'resid_pre')
            elif intervene.startswith('shape'):
                for img_pos in vocab.image_patches[intervene]:
                    alter_cache_i1 = replace_cache(alter_cache_i1, ctx2_cache,
                                         img_pos, img_pos, num_layers, 'resid_pre')
                

        # target_dependent_mask = torch.zeros(
        #     (batch_size, num_layers, ctx1_inputs.input_ids.shape[1]+(num_image_patches-1)), dtype=bool
        # )
        target_dependent_mask = torch.zeros(
            (batch_size, num_layers, ctx1_inputs.input_ids.shape[1]), dtype=bool
        )
        dependent_start = ctx1_aligned_indices[0]['context'][1]
        target_dependent_mask[:, :, dependent_start:] = True
        source_mask = ~target_dependent_mask

        fwd_hooks=[(
                get_act_name('resid_pre', layer),
                partial(patch_masked_residue,mask=source_mask[:,layer,:],source_cache=alter_cache_i1)
            ) for layer in range(num_layers)]
        
        for shape_id in range(4):
            ctx1_qn,_,ctx2_qn,_ = zip(*vocab.get_samples(list(range(start_idx,end_idx)),shape_id))
            inputs = model.processor(images=ctx1_img_list, text=ctx1_qn, return_tensors='pt').to(device)
            logits = model.run_with_hooks(**inputs, fwd_hooks=fwd_hooks)

            for i, (ctx1, ctx2) in enumerate(zip(ctx1_list, ctx2_list)):
                print(ctx1_qn[0])
                idx = start_idx+i
                item_logits, score = evaluate_factorizabilty(logits[i], ctx1, ctx2, model.processor.tokenizer)
                log_probs_matrix[shape_id][idx] = item_logits
                counts[shape_id] += score
    
            print(idx,counts)

        del ctx1_img_list, ctx2_img_list, ctx1_inputs, ctx2_inputs, inputs
        del ctx1_cache, ctx2_cache, alter_cache_i1, logits
        gc.collect()

    log_probs = log_probs_matrix.mean(dim=-2)

    print('__________________Accuracy_________________')
    print(counts)

    print('__________________Mean Log Probs_________________')
    print(log_probs)

    return dict(accuracy=counts, log_probs=log_probs_matrix)



def run_factorizability_llava_ov(args):

    dataset_name = 'shapes'
    im_flip = False

    save_path_temp = 'outputs/factorizability_{}.pth'

    vocab = ShapesItems(csv_path='dataset/factorizability.csv', 
                        img_path='', c_mapping=False, s_mapping=True, flip=False)
    llava_ov_deets = get_llava_ov_details(w=3,  im_flip=im_flip)
    vocab.set_details(**llava_ov_deets, flip=False)

    num_samples = len(vocab.rows)

    device = torch.device('cuda')
    # device = torch.device('cpu')
    model = fetch_llava_ov_model(cache_dir=args.cache_dir, device='cuda',num_devices=1)
    model.to(device=device)
    num_layers = model.hooked_language_model.cfg.n_layers

    no_intervene = [[]]
    item_intervenes = [['item1'], ['item2'], ['item1','item2']]
    patch_intervenes = [['shape1'], ['shape2'], ['shape1','shape2']]
    color_intervenes = [['color1'], ['color2'], ['color1','color2']]
    all_intervenes = [
        *no_intervene, *patch_intervenes, *item_intervenes, *color_intervenes
    ]


    for intervention in all_intervenes:
        print(intervention)
        save_path = save_path_temp.format('_'.join(intervention))
        print(save_path)
        result = intervene(model=model, vocab=vocab, num_samples=num_samples, num_layers=num_layers, 
                       device=device,  intervene_list=intervention, bs=1, num_image_patches=(27*27)+(28*27))
        torch.save(result, save_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, help='HF cache dir path')

    args = parser.parse_args()

    run_factorizability_llava_ov(args)
