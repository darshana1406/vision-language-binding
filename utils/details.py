from utils.helper import *


def get_llava_ov_details(w=0, im_flip=False):

    template = """<|im_start|>user <image>\n
Answer the question based on the provided image and the context below. Keep the answer short.

Context:{context}

Question: Which item does the {qn_shape} contain?<|im_end|><|im_start|>assistant
Answer: The {ans_shape} contains item"""

    context_template = 'The {color} object contains item {letter}.'

    if im_flip:
        shape1_list = [(97,101+1),(124,128+1),(151,155+1),(178,182+1),(205,209+1)]
        shape2_list = [(164,168+1),(191,195+1),(218,222+1),(245,249+1),(272,276+1)]
    else:
        shape1_list = [(87,91+1),(114,118+1),(141,145+1),(168,172+1),(195,199+1)]
        shape2_list = [(182,186+1),(209,213+1),(236,240+1),(263,267+1),(290,294+1)]
    if im_flip:
        images_root = 'dataset/images_large_384_flipped/'
    else:
        images_root = 'dataset/images_large_384/'


    top = shape1_list[0][0]
    bottom = shape1_list[-1][0]
    size = shape1_list[0][1]-top
    pad_shape1_list = [(s-w,e+w) for s,e in shape1_list]
    for i in range(1,w+1):
        top_row = top - 27*i
        pad_shape1_list.append((top_row-w,top_row+size+w))
        bottom_row = bottom + 27*i
        pad_shape1_list.append((bottom_row-w,bottom_row+size+w))
    shape1_list = pad_shape1_list

    top = shape2_list[0][0]
    bottom = shape2_list[-1][0]
    size = shape2_list[0][1]-top
    pad_shape2_list = [(s-w,e+w) for s,e in shape2_list]
    for i in range(1,w+1):
        top_row = top - 27*i
        pad_shape2_list.append((top_row-w,top_row+size+w))
        bottom_row = bottom + 27*i
        pad_shape2_list.append((bottom_row-w,bottom_row+size+w))
    shape2_list = pad_shape2_list

    any_res_shape1_list, any_res_shape2_list = [], []
    any_res_offset = 27**2
    for s,e in shape1_list:
        num_end_toks = s//27 + any_res_offset
        any_res_shape1_list.append((s+num_end_toks, e+num_end_toks))
    for s,e in shape2_list:
        num_end_toks = s//27 + any_res_offset
        any_res_shape2_list.append((s+num_end_toks, e+num_end_toks))

    image_offset = 3
    shape1_patches = [Substring(*tup)+image_offset for tup in shape1_list+any_res_shape1_list ] 
    shape2_patches = [Substring(*tup)+image_offset for tup in shape2_list+any_res_shape2_list ]

    return dict(
        template = template,
        context_template = context_template,
        image_offset = image_offset,
        shape1_patches = shape1_patches,
        shape2_patches = shape2_patches,
        img_path = images_root
    )



def get_llava_ov_details_mean_est(w=0):

    template = """<|im_start|>user <image>\n
Answer the question based on the provided image and the context below. Keep the answer short.

Context:{context}

Question: Which item does the {qn_shape} contain?<|im_end|><|im_start|>assistant
Answer: The {ans_shape} contains item"""

    context_template = 'The {color} object contains item {letter}.'


    shape1_list = [(87,91+1),(114,118+1),(141,145+1),(168,172+1),(195,199+1)]
    shape2_list = [(182,186+1),(209,213+1),(236,240+1),(263,267+1),(290,294+1)]

    images_root = 'dataset/images_large_384_other/'


    top = shape1_list[0][0]
    bottom = shape1_list[-1][0]
    size = shape1_list[0][1]-top
    pad_shape1_list = [(s-w,e+w) for s,e in shape1_list]
    for i in range(1,w+1):
        top_row = top - 27*i
        pad_shape1_list.append((top_row-w,top_row+size+w))
        bottom_row = bottom + 27*i
        pad_shape1_list.append((bottom_row-w,bottom_row+size+w))
    shape1_list = pad_shape1_list

    top = shape2_list[0][0]
    bottom = shape2_list[-1][0]
    size = shape2_list[0][1]-top
    pad_shape2_list = [(s-w,e+w) for s,e in shape2_list]
    for i in range(1,w+1):
        top_row = top - 27*i
        pad_shape2_list.append((top_row-w,top_row+size+w))
        bottom_row = bottom + 27*i
        pad_shape2_list.append((bottom_row-w,bottom_row+size+w))
    shape2_list = pad_shape2_list

    any_res_shape1_list, any_res_shape2_list = [], []
    any_res_offset = 27**2
    for s,e in shape1_list:
        num_end_toks = s//27 + any_res_offset
        any_res_shape1_list.append((s+num_end_toks, e+num_end_toks))
    for s,e in shape2_list:
        num_end_toks = s//27 + any_res_offset
        any_res_shape2_list.append((s+num_end_toks, e+num_end_toks))

    image_offset = 3
    shape1_patches = [Substring(*tup)+image_offset for tup in shape1_list+any_res_shape1_list ] 
    shape2_patches = [Substring(*tup)+image_offset for tup in shape2_list+any_res_shape2_list ]

    return dict(
        template = template,
        context_template = context_template,
        image_offset = image_offset,
        shape1_patches = shape1_patches,
        shape2_patches = shape2_patches,
        img_path = images_root
    )


