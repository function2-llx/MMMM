from pathlib import Path

from jsonargparse import class_from_function
from transformers import LlamaTokenizer

class MMMMTokenizer(LlamaTokenizer):
    def __init__(self, *args, use_seg_token: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_vocab_size = self.vocab_size
        # TODO: can we simplify these boilerplate codes while keeping code completion?
        self.bop_token = '<p>'
        self.eop_token = '</p>'
        self.usr_token = '<USR>'
        self.sys_token = '<SYS>'
        self.add_tokens(
            [self.bop_token, self.eop_token, self.usr_token, self.sys_token],
            True,
        )
        self.bop_token_id, self.eop_token_id, self.usr_token_id, self.sys_token_id = self.convert_tokens_to_ids([
            self.bop_token, self.eop_token, self.usr_token, self.sys_token,
        ])
        self.use_seg_token = use_seg_token
        if use_seg_token:
            self.seg_token = '<SEG>'
            self.add_tokens(self.seg_token, special_tokens=True)
            self.seg_token_id: int = self.convert_tokens_to_ids(self.seg_token)

    @classmethod
    def build(cls, hf_model_path: Path, use_seg_token: bool = True):
        # no type hint (https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/tokenization_utils_base.py#L1827)
        # will cause jsonargparse fail (https://github.com/omni-us/jsonargparse/issues/454).
        return cls.from_pretrained(hf_model_path, use_seg_token=use_seg_token)

build = class_from_function(MMMMTokenizer.build, MMMMTokenizer)
