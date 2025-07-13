from dataclasses import dataclass
import bisect

@dataclass(frozen=True)
class Substring:
    start: int
    end: int

    def __iter__(self):
        return iter((self.start, self.end))

    def __getitem__(self, key):
        if key == 0:
            return self.start
        else:
            return self.end

    def to_slice(self):
        return slice(self.start, self.end)

    def __add__(self, num):
        return Substring(self.start + num, self.end + num)
    


def recursify(func, dtype=Substring, pred=None):
    if pred is None:
        pred = lambda x: isinstance(x, dtype)

    def wrapper(indices, *args, **kwargs):
        if pred(indices):
            return func(indices, *args, **kwargs)
        elif isinstance(indices, dict):
            return {
                key: wrapper(value, *args, **kwargs) for key, value in indices.items()
            }
        elif isinstance(indices, list):
            return [wrapper(value, *args, **kwargs) for value in indices]
        else:
            raise Exception(f"Unexpected type {type(indices)}")

    return wrapper



def align_token_indices(tokenizer, prompt, index, num_image_patches=576):
    tokenized = tokenizer(prompt, return_offsets_mapping=True)
    inputs, offset_mapping = tokenized['input_ids'], tokenized['offset_mapping']

    @recursify
    def align(pos, num_image_patches):
        start, end = pos
        start = bisect.bisect_right([x for x, _ in offset_mapping], start) - 1
        end = bisect.bisect_right([x for x, _ in offset_mapping], end-1) - 1
        start += num_image_patches-1
        end += num_image_patches-1
        return Substring(start, end+1)
    
    aligned_index = align(index, num_image_patches=num_image_patches)
    return inputs, aligned_index