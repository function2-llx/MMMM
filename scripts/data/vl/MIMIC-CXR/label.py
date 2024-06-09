import shutil

import cytoolz
import orjson
import torch
from torch.types import Device
from tqdm import tqdm
from transformers import BertTokenizerFast

from CheXbert.constants import CONDITIONS
from CheXbert.models.bert_encoder import bert_encoder

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT

CHEXBERT_PATH = '/data/chexbert/chexbert.pth'

class Labeler:
    def __init__(self, device: Device = 'cuda'):
        self.device = device
        model = bert_encoder(logits=True)
        checkpoint = torch.load(CHEXBERT_PATH, map_location='cpu')
        state_dict = {
            key[7:]: weight
            for key, weight in checkpoint['model_state_dict'].items()
        }
        model.load_state_dict(state_dict)
        model = model.to(device=device)
        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    @torch.inference_mode()
    def __call__(self, text: list[str]) -> torch.BoolTensor:
        """
        Returns:
            binary classification results, (batch size, C = 14)
        """
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True).to(device=self.device)
        out: list[torch.Tensor] = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # classes: present, absent, unknown, blank for 12 conditions + support devices
        # classes: yes, no for the 'no finding' observation
        pred = torch.stack([prob.argmax(dim=-1) for i, prob in enumerate(out)], dim=-1)
        # Positive and Uncertain in CLASS_MAPPING
        pred = (pred == 1) | (pred == 3)
        return pred

NO_FINDING_IDX = CONDITIONS.index('No Finding')

labels = [
    ('Atelectasis', 'atelectasis'),
    ('Cardiomegaly', 'cardiomegaly'),
    ('Consolidation', 'pulmonary consolidation'),
    ('Edema', 'pulmonary edema'),
    ('Enlarged Cardiomediastinum', 'widened mediastinum'),
    ('Fracture', 'rib fracture'),
    ('Lung Lesion', 'lung nodule'),  # looks like they refer to the same thing in MIMIC-CXR
    ('Lung Opacity', 'pulmonary opacification'),
    ('Pleural Effusion', 'pleural effusion'),
    # ('Pleural Other'),
    ('Pneumonia', 'pneumonia'),
    ('Pneumothorax', 'pneumothorax'),
]
labels_dict = dict(labels)
for key in labels_dict:
    assert key in CONDITIONS

def _get_data(dataset: str, split: str, suffix: str) -> list[dict]:
    if not (dst := PROCESSED_VL_DATA_ROOT / dataset / f'archive/{split}-processed-{suffix}.json').exists():
        shutil.copy(PROCESSED_VL_DATA_ROOT / dataset / f'{split}-processed.json', dst)
    return orjson.loads(dst.read_bytes())

def main():
    labeler = Labeler()
    for split in ['test', 'train']:
        data = _get_data('MIMIC-CXR', split, 'to-label')
        item_len = []
        for idx, item in enumerate(tqdm(data, desc='tokenizing')):
            tokens = labeler.tokenizer.tokenize(item['processed_report'])
            item_len.append(len(tokens))
        data_iter_idx = sorted(range(len(data)), key=lambda idx: item_len[idx])
        batch_size = 32
        for batch_idx in tqdm(
            cytoolz.partition_all(batch_size, data_iter_idx), total=(len(data) - 1) // batch_size + 1,
        ):
            texts = [data[item_idx]['processed_report'] for item_idx in batch_idx]
            preds = labeler(texts)
            for data_idx, pred in zip(batch_idx, preds):
                item = data[data_idx]
                if pred[NO_FINDING_IDX]:
                    item['anomaly_pos'] = []
                    item['anomaly_neg'] = [name for _, name in labels]
                else:
                    item['anomaly_pos'] = []
                    item['anomaly_neg'] = []
                    for c, condition in enumerate(CONDITIONS):
                        if name := labels_dict.get(condition):
                            item['anomaly_pos' if pred[c] else 'anomaly_neg'].append(name)
        (PROCESSED_VL_DATA_ROOT / 'MIMIC-CXR' / f'{split}-processed.json').write_bytes(
            orjson.dumps(data, option=orjson.OPT_INDENT_2),
        )

if __name__ == '__main__':
    main()
