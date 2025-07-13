import torch

def median_calib_acc(logits: torch.Tensor):
    num_shapes = logits.shape[0]
    num_attr = logits.shape[-1]
    scores = torch.zeros((num_shapes, num_attr))
    mean_logits = logits.reshape((-1,4)).quantile(0.5, dim=-2)

    logits -= mean_logits

    for i in range(num_shapes):
        val, ind = logits[i].max(dim=-1)
        for j in range(num_attr):
            scores[i][j] = (ind == j).sum()

    return scores


def evaluate_factorizabilty(logits, ctx1, ctx2, tokenizer):
    score = torch.zeros(4)
    items = [ctx1['items']['item1'], ctx1['items']['item2'],
             ctx2['items']['item1'], ctx2['items']['item2'],
            ]

    item_tokens = [ tokenizer.encode(item, add_special_tokens=False)[0] for item in items] 
    logits = logits.squeeze()[-1]
    item_logits = logits[item_tokens]
    item_log_probs = item_logits - logits.logsumexp(dim=-1, keepdim=False)

    val, ind = item_log_probs.max(-1)
    score[ind.item()] = 1.

    return item_log_probs, score


def evaluate(logits, ctx1, tokenizer):
    score = torch.zeros(2)
    items = [ctx1['items']['item1'], ctx1['items']['item2']]
    
    item_tokens = [ tokenizer.encode(item, add_special_tokens=False)[0] for item in items] 
    logits = logits.squeeze()[-1]
    item_logits = logits[item_tokens]
    item_log_probs = item_logits - logits.logsumexp(dim=-1, keepdim=False)

    val, ind = item_log_probs.max(-1)
    score[ind.item()] = 1.

    # if max(item_logits) == item_logits[item_id]:
    #     score = 1

    return item_log_probs, score