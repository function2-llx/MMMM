from jsonargparse import class_from_function
from transformers import LlamaTokenizer

class MMMMTokenizer(LlamaTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_open = '<MASK>'
        self.mask_close = '</MASK>'
        self.inst_open = '<INST>'
        self.inst_close = '</INST>'

        self.add_tokens([self.mask_open, self.mask_close, self.inst_open, self.inst_close], True)
        self.mask_open_id, self.mask_close_id = self.convert_tokens_to_ids([self.mask_open, self.mask_close])
        self.inst_open_id, self.inst_close_id = self.convert_tokens_to_ids([self.inst_open, self.inst_close])

from_pretrained = class_from_function(MMMMTokenizer.from_pretrained, MMMMTokenizer)
