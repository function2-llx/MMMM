from pathlib import Path

from jsonargparse import class_from_function
import torch
from transformers import LlamaTokenizer

class MMMMTokenizer(LlamaTokenizer):
    sys_token = '<sys>'
    sys_token_id: int
    usr_token = '<usr>'
    usr_token_id: int
    # enable grounding
    grd_token = '<grd>'
    grd_token_id: int
    # disable grounding
    ngrd_token = '<ngrd>'
    ngrd_token_id: int
    # begin of phrase
    bop_token = '<p>'
    bop_token_id: int
    eop_token = '</p>'
    eop_token_id: int
    # begin of negative phrase, not actually used by model
    bonp_token = '<np>'
    bonp_token_id: int
    eonp_token = '</np>'
    eonp_token_id: int

    def __init__(self, *args, use_seg_token: bool, share_seg_token: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_vocab_size = self.vocab_size

        special_token_names = [*map(
            lambda name: f'{name}_token',
            ['sys', 'usr', 'grd', 'ngrd', 'bop', 'eop', 'bonp', 'eonp'],
        )]
        special_tokens = [*map(self.__getattribute__, special_token_names)]
        self.add_tokens(special_tokens, special_tokens=True)
        special_token_ids = self.convert_tokens_to_ids(special_tokens)
        for token_name, special_token_id in zip(special_token_names, special_token_ids):
            setattr(self, f'{token_name}_id', special_token_id)

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

    def wrap_name(self, name: str, neg: bool = False):
        if neg:
            bop_token, eop_token = self.bonp_token, self.eonp_token
        else:
            bop_token, eop_token = self.bop_token, self.eop_token
        ret = f'{bop_token} {name}{eop_token}'
        if self.use_seg_token:
            if self.share_seg_token:
                seg_token = self.seg_token
            else:
                seg_token = self.seg_tokens[self.class_to_idx[name]]
            ret = f'{ret} {seg_token}'
        return ret

    def wrap_name_neg(self, name: str):
        return self.wrap_name(name, neg=True)

build = class_from_function(MMMMTokenizer.build, MMMMTokenizer)
