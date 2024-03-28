from pathlib import Path

from jsonargparse import class_from_function
import torch
from transformers import LlamaTokenizer

class MMMMTokenizer(LlamaTokenizer):
    def __init__(self, *args, use_seg_token: bool, share_seg_token: bool, **kwargs):
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
        self.share_seg_token = share_seg_token
        if use_seg_token:
            if share_seg_token:
                self.seg_token = '[SEG]'
                self.add_tokens(self.seg_token, special_tokens=True)
                self.seg_token_id: int = self.convert_tokens_to_ids(self.seg_token)
            else:
                self.seg_token_id_start = len(self)
                self.seg_tokens = [f'[SEG-{i}]' for i in range(500)]
                self.add_tokens(self.seg_tokens, True)

    @classmethod
    def build(cls, hf_model_path: Path, use_seg_token: bool = False, share_seg_token: bool = True):
        # no type hint (https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/tokenization_utils_base.py#L1827)
        # will cause jsonargparse fail (https://github.com/omni-us/jsonargparse/issues/454).
        return cls.from_pretrained(
            hf_model_path, use_seg_token=use_seg_token, share_seg_token=share_seg_token,
        )

    def create_seg_token_mask(self, token_ids: torch.LongTensor):
        """
        Args:
            token_ids: the (possibly shifted) token ids to find seg tokens
        """
        if self.use_seg_token:
            if self.share_seg_token:
                return token_ids == self.seg_token_id
            else:
                return token_ids >= self.seg_token_id_start
        else:
            return token_ids == self.eop_token_id

    def _parse_targets(self, token_ids: list[int]) -> list[str] | None:
        ret = []
        last_bop: int | None = None
        for i, token_id in enumerate(token_ids):
            match token_id:
                case self.bop_token_id:
                    if last_bop is not None:
                        return None
                    last_bop = i
                case self.eop_token_id:
                    if last_bop is None:
                        return None
                    ret.append(self.decode(token_ids[last_bop + 1:i - 1]))
                    last_bop = None
        return ret

    def parse_targets(self, token_ids: torch.LongTensor) -> list[list[str] | None]:
        return [
            self._parse_targets(token_ids[i].tolist())
            for i in range(token_ids.shape[0])
        ]

    def build_classes_index(self, names: set[str]):
        """This method is useful only when not self.share_seg_token"""
        self.class_to_idx = {name: i for i, name in enumerate(sorted(names))}

    def wrap_name(self, name: str):
        ret = f'{self.bop_token} {name} {self.eop_token}'
        if self.use_seg_token:
            if self.share_seg_token:
                seg_token = self.seg_token
            else:
                seg_token = self.seg_tokens[self.class_to_idx[name]]
            ret = f'{ret} {seg_token}'
        return ret

build = class_from_function(MMMMTokenizer.build, MMMMTokenizer)
