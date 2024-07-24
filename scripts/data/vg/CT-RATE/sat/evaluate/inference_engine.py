from pathlib import Path

from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
import monai.transforms as mt
import orjson
import torch
from tqdm import tqdm

from luolib.utils import save_pt_zst
from luolib.utils.misc import min_stem

from model.maskformer import Maskformer

def save_prediction(
    prediction: torch.Tensor,
    save_path: Path,
    labels: list[str],
):
    assert save_path.name.endswith('.pt.zst')
    save_path.with_name(f'{min_stem(save_path)}.json').write_bytes(orjson.dumps(labels, option=orjson.OPT_INDENT_2))
    # save prediction masks at last for completeness checking
    save_pt_zst(prediction.cpu(), save_path, atomic=True)

def inference(
    model: Maskformer,
    text_encoder,
    test_loader,
    sw_batch_size: int,
):
    """
    model should be setup by fabric to make autocast work automatically
    """
    model.eval()
    text_encoder.eval()
    with torch.inference_mode():
        # gaussian kernel to accumulate predcition
        for batch in tqdm(test_loader, desc='iterate test dataloader', dynamic_ncols=True):
            try:
                # data loading
                meta = batch['meta']
                split_labels = batch['split_queries']
                split_n1n2 = batch['split_n1n2']
                labels = batch['labels']
                modality = batch['modality']

                queries_ls = []
                for labels_ls, n1n2 in zip(split_labels, split_n1n2):  # convert list of texts to list of embeds
                    queries_ls.append(text_encoder(labels_ls, modality))
                prob = sliding_window_inference(
                    batch['image'][None],
                    (288, 288, 96),
                    sw_batch_size,
                    lambda patch: model(queries_ls, patch).sigmoid(),
                    overlap=0.5,
                    mode='gaussian',
                    # progress=True,
                )[0]
                original_affine = meta['original_affine']
                prob = mt.SpatialResample(dtype=None).__call__(
                    MetaTensor(prob, meta['affine']),
                    original_affine,
                    meta['spatial_shape'],
                    padding_mode='zeros',
                )
                prediction = (prob > 0.5).bool()
                save_prediction(prediction, batch['save_path'], labels)
            except Exception as e:
                print(e)
                if save_path := batch.get('save_path'):
                    print('save_path:', save_path)
                else:
                    print('`save_path` not found in batch, really?')
