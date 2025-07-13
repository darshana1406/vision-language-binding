import pandas as pd
from string import Formatter
import bisect
import ast
from os.path import join

from utils.helper import Substring, recursify



class ShapesItems():

    def __init__(self, csv_path, img_path, flip=False, c_mapping=True, s_mapping=True) -> None:

        self.flip = flip
        self.c_mapping = c_mapping
        self.s_mapping = s_mapping

        self.data = pd.read_csv(csv_path)
        self.data.image1 = self.data.image1.apply(ast.literal_eval)
        self.data.image2 = self.data.image2.apply(ast.literal_eval)
        self.data.img1_items = self.data.img1_items.apply(ast.literal_eval)
        self.data.img2_items = self.data.img2_items.apply(ast.literal_eval)
        self.rows = [row for i, row in self.data.iterrows()]

        self.images_path = img_path

        self.context_template = 'The {color} object contains item {letter}.'


        self.template = None

        self.image_offset = None
        shape1_patches = None
        shape2_patches = None
        self.image_patches = dict(
            shape1 = shape1_patches,
            shape2 = shape2_patches
    )

            

    def set_details(self, context_template, template, image_offset, shape1_patches, shape2_patches, flip, img_path):
        self.context_template = context_template
        self.template = template
        self.images_path = img_path
        self.image_offset = image_offset
        self.flip = flip

        self.image_patches = dict(
            shape1 = shape1_patches,
            shape2 = shape2_patches
        ) 
    

        
    def recursify(func, dtype=Substring, pred=None):
        if pred is None:
            pred = lambda x: isinstance(x, Substring) or isinstance(x, int)

        def wrapper(self, indices, *args, **kwargs):
            if pred(indices):
                return func(self, indices, *args, **kwargs)
            elif isinstance(indices, dict):
                return {
                    key: wrapper(self, value, *args, **kwargs) for key, value in indices.items()
                }
            elif isinstance(indices, list):
                return [wrapper(self, value, *args, **kwargs) for value in indices]
            else:
                raise Exception(f"Unexpected type {type(indices)}")

        return wrapper


    @recursify
    def get_sample_dicts(self, index, flip_item=False, flip_img=False):

        # if self.c_mapping:
        #     c_map = lambda c: 'bright '+c if not (c in ['purple','cyan']) else c
        # else:
        #     c_map = lambda c: c

        # if self.s_mapping:
        #     s_map = lambda s: 'can' if s == 'cylinder' else s
        # else:
        #     s_map = lambda s: s
        if self.c_mapping:
            c_map = lambda c: 'bright '+c if not (c in ['purple','cyan']) else c
        else:
            c_map = lambda c: c

        if self.s_mapping:
            def smap(s):
                if s == 'cylinder':
                    return 'can'
                elif s == 'pyramid':
                    return '  pyramid'
                elif s == 'prism':
                    return '  prism'
                else:
                    return s
            s_map = smap
        else:
            s_map = lambda s: s

        row = self.rows[index]
        context_s1_image = dict(
            color1 = c_map(row['image1'][0]),shape1 = s_map(row['image1'][1]),
            color2 = c_map(row['image1'][2]),shape2 = s_map(row['image1'][3])
        )
        if flip_item:
            context_s1_items = dict(item1 = row['img1_items'][1],item2 = row['img1_items'][0])
        else:
            context_s1_items = dict(item1 = row['img1_items'][0],item2 = row['img1_items'][1])
        context_s2_image = dict(
            color1 = c_map(row['image2'][0]),shape1 = s_map(row['image2'][1]),
            color2 = c_map(row['image2'][2]),shape2 = s_map(row['image2'][3])
        )
        if flip_item:
            context_s2_items = dict(item1 = row['img2_items'][1],item2 = row['img2_items'][0])
        else:
            context_s2_items = dict(item1 = row['img2_items'][0],item2 = row['img2_items'][1])

        if flip_img:
            path_t = 'images/{}_{}_{}_{}.png'
          
            context_s1 = dict(
            image_path = join(self.images_path,path_t.format(
                row['image1'][2],row['image1'][3], 
                row['image1'][0],row['image1'][1])),
            objects = context_s1_image,
            items = context_s1_items
        )
            context_s2 = dict(
                image_path = join(self.images_path,path_t.format(
                    row['image2'][2],row['image2'][3], 
                    row['image2'][0],row['image2'][1])),
                objects = context_s2_image,
                items = context_s2_items
            )
        else:
            context_s1 = dict(
                image_path = join(self.images_path,row['img1_path']),
                objects = context_s1_image,
                items = context_s1_items
            )
            context_s2 = dict(

                image_path = join(self.images_path,row['img2_path']),
                objects = context_s2_image,
                items = context_s2_items
            )

        if self.flip:
            return context_s2, context_s1
        return context_s1, context_s2
    

    def template_format(self, template, format_dict, offset=0):
        ind_dict = {}
        cur_offset = 0
        for literal_text, field_name, format_spec, conversion in Formatter().parse(template):
            if field_name is not None:
                value = format_dict[field_name]
                start = len(literal_text) + cur_offset
                end = start + len(value)
                ind_dict[field_name] = Substring(start+offset,end+offset)
                cur_offset = end

        format_str = template.format(**format_dict)

        return format_str,ind_dict
    


    def get_context(self, context, shape, order_flip=False):
        context_p1, ind_dict_p1 = self.template_format(
            self.context_template,
            dict(color=context['objects']['color1'], letter=context['items']['item1'])
        )
        context_p2, ind_dict_p2 = self.template_format(
            self.context_template,
            dict(color=context['objects']['color2'], letter=context['items']['item2'])
        )

        if order_flip:
            context_p1, context_p2 = context_p2, context_p1
            ind_dict_p1, ind_dict_p2 = ind_dict_p2, ind_dict_p1
            # assert ind_dict_p1 == ind_dict_p2 True


        joined_context = ' '.join([context_p1, context_p2])
        for key in ind_dict_p2:
            ind_dict_p2[key] += len(context_p1) + 1

        text_prompt, text_ind_dict = self.template_format(
            self.template, 
            dict(context=joined_context, qn_shape=shape, ans_shape=shape)
            )

        for key in ind_dict_p1:
            ind_dict_p1[key] += text_ind_dict['context'][0]
        for key in ind_dict_p2:
            ind_dict_p2[key] += text_ind_dict['context'][0]

        full_ind_dict = dict(
            color1 = ind_dict_p1['color'],
            item1 = ind_dict_p1['letter'],
            color2 = ind_dict_p2['color'],
            item2 = ind_dict_p2['letter'],
            context = text_ind_dict['context'],
            qn_shape = text_ind_dict['qn_shape'],
            ans_shape = text_ind_dict['ans_shape']
        )

        return text_prompt, full_ind_dict

    @recursify
    def get_samples(self, index, shape_id, flip_item=False, flip_img=False, order_flip=False):

        context_s1, context_s2 = self.get_sample_dicts(index, flip_item=flip_item, flip_img=False)
        if shape_id == 0:
            shape = context_s1['objects']['shape1']
        elif shape_id == 1:
            shape = context_s1['objects']['shape2']
        elif shape_id == 2:
            shape = context_s2['objects']['shape1']
        elif shape_id == 3:
            shape = context_s2['objects']['shape2']
        else:
            shape = shape_id

        s1_text, s1_index = self.get_context(context_s1, shape, order_flip=order_flip)
        s2_text, s2_index = self.get_context(context_s2, shape, order_flip=order_flip)

        return s1_text, s1_index, s2_text, s2_index
    


    

