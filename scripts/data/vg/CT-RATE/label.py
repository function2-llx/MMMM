from typing import TYPE_CHECKING

import einops
import numpy as np
import torch
import torch.nn.functional as nnf
from jsonargparse import ArgumentParser
from lightning import Fabric
from transformers import AutoModel, AutoTokenizer

from luolib.data.utils import list_data_collate
from luolib.types import PathLike
from luolib.utils import load_pt_zst

from monai.inferers import sliding_window_inference
from mmmm.data.defs import PROCESSED_VG_DATA_ROOT

if TYPE_CHECKING:
    import _stub.SegVol.model_segvol_single as _segvol

vg_data_dir = PROCESSED_VG_DATA_ROOT / 'CT-RATE'

fabric = Fabric()

# target_tax = get_target_tax()
clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
segvol: '_segvol.SegVolModel' = AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True, test_mode=True)
segvol.model.text_encoder.tokenizer = clip_tokenizer
segvol.eval()
segvol = fabric.setup(segvol)

@torch.no_grad()
def _dice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 2 * (x * y).sum() / (x.sum() + y.sum())

def intensity_norm(ct_narray):
    ct_voxel_ndarray = ct_narray.copy()
    ct_voxel_ndarray = ct_voxel_ndarray.flatten()
    thred = np.mean(ct_voxel_ndarray)
    voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
    upper_bound = np.percentile(voxel_filtered, 99.95)
    lower_bound = np.percentile(voxel_filtered, 00.05)
    mean = np.mean(voxel_filtered)
    std = np.std(voxel_filtered)
    ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
    ct_narray = (ct_narray - mean) / max(std, 1e-8)
    return ct_narray

def process_ct(path: PathLike):
    """adapted from `SegVolProcessor.preprocess_ct_gt`"""
    item = {}
    if str(path).endswith('.pt.zst'):
        ct_voxel_ndarray: np.ndarray = load_pt_zst(path).numpy()
        # make SegVol happy
        # ct_voxel_ndarray = einops.rearrange(ct_voxel_ndarray, 'c d h w -> c w h d')
        ct_voxel_ndarray = einops.rearrange(ct_voxel_ndarray, 'c d h w -> c h w d')
    else:
        import monai.transforms as mt
        loader = mt.Compose([
            mt.LoadImage(ensure_channel_first=True),
            mt.Orientation('RAS'),
        ])
        ct_voxel = loader(path)
        ct_voxel_ndarray = ct_voxel.numpy()
    ct_voxel_ndarray = intensity_norm(ct_voxel_ndarray)
    item['image'] = ct_voxel_ndarray.astype(np.float32)
    # dummy label
    item['label'] = np.zeros_like(ct_voxel_ndarray, dtype=np.int16)

    return item['image'], item['label']

def logits2roi_coor(spatial_size, logits_global_single):
    # crop predict
    pred_global_single = torch.sigmoid(logits_global_single) > 0.5
    ## get all pos idx
    nonzero_indices = torch.nonzero(pred_global_single)
    if nonzero_indices.shape[0] == 0:
        return None, None, None, None, None, None
    ## get boundary
    min_d, max_d = nonzero_indices[:, 0].min(), nonzero_indices[:, 0].max()
    min_h, max_h = nonzero_indices[:, 1].min(), nonzero_indices[:, 1].max()
    min_w, max_w = nonzero_indices[:, 2].min(), nonzero_indices[:, 2].max()
    ## padding
    crop_d, crop_h, crop_w = max_d - min_d + 1, max_h - min_h + 1, max_w - min_w + 1,
    window_d, window_h, window_w = spatial_size
    padding_d, padding_h, padding_w = max(0, window_d-crop_d), max(0, window_h-crop_h), max(0, window_w-crop_w)
    global_d, global_h, global_w = logits_global_single.shape
    min_d = max(0, min_d - int(padding_d)//2)
    min_h = max(0, min_h - int(padding_h)//2)
    min_w = max(0, min_w - int(padding_w)//2)
    max_d = min(global_d, max_d + int(padding_d)//2)
    max_h = min(global_h, max_h + int(padding_h)//2)
    max_w = min(global_w, max_w + int(padding_w)//2)
    return min_d, min_h, min_w, max_d, max_h, max_w

def forward_test(
    self: '_segvol.SegVolModel',
    image,
    zoomed_image,
    text_prompt: list[str],
    use_zoom=True,
):
    # assert image.shape[0] == 1 and zoomed_image.shape[0] == 1, 'batch size should be 1'
    # volume_shape = image[0][0].shape
    # with torch.no_grad():
    #     logits_global_single = self.model(zoomed_image, text=text_prompt)
    # logits_global_single = nnf.interpolate(logits_global_single.cpu(), size=volume_shape, mode='nearest')
    # if not use_zoom:
    #     return logits_global_single
    #
    # min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(self.config.spatial_size, logits_global_single[0][0])
    # if min_d is None:
    #     print('Fail to detect foreground!')
    #     return logits_global_single

    # Crop roi
    # image_single_cropped = image[:, :, min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]
    ## inference
    with torch.inference_mode():
        logits_single_cropped = sliding_window_inference(
            image,
            self.config.spatial_size,
            sw_batch_size=4,
            predictor=self.model,
            overlap=0.5,
            text=text_prompt,
            # mode='gaussian',
            progress=True,
        )
        logits_single_cropped = logits_single_cropped.cpu().squeeze()
    return logits_single_cropped[None, None]
    # logits_global_single[:, :, min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1] = logits_single_cropped
    # return logits_global_single

def infer_case(path: PathLike):
    for category in ['left kidney', 'spleen', 'pancreas', 'liver', 'left lung', 'right lung', 'left lung upper lobe', 'heart']:
        image, label = process_ct(path)
        image = np.rot90(image, k=2, axes=(1, 2))
        item: dict = segvol.processor.zoom_transform(image, label)
        categories = [category]
        item = list_data_collate([item])
        item = fabric.to_device(item)
        logits_mask = forward_test(
            segvol,
            image=item['image'],
            zoomed_image=item['zoom_out_image'],
            text_prompt=categories,
            use_zoom=True
        )
        pred = logits_mask.sigmoid() > 0.5
        from luolib.utils import IndexTracker
        IndexTracker(
            item['image'][0, 0],
            pred[0, 0],
            choose_max=True,
            title=category,
        )
    # IndexTrackerBinary(
    #     item['image'][0, 0],
    #     pred[0],
    # )
    # pass

# def process(split: str):
#     # data_list = orjson.loads(vg_data_dir / f'{split}.json')
#     'data/processed/vision-language/CT-RATE/image/train_19756_a/train_19756_a_2.pt.zst'

def main():
    from mmmm.models.sam.model import AlignSam
    parser = ArgumentParser()
    parser.add_class_arguments(AlignSam, 'model', )
    # infer_case('data/processed/vision-language/CT-RATE/image/train_19756_a/train_19756_a_2.pt.zst')
    # infer_case('train_19756_a_2.nii.gz')
    infer_case('data/train_19756_a_1.nii.gz')
    exit(0)
    parser = ArgumentParser()
    parser.add_argument('-c', action=ActionConfigFile)
    parser.add_class_arguments(AlignSam, 'model')
    parser.add_argument('--ckpt_path', type=Path | None)
    # parser.add_argument('--max_vision_tokens', type=int, required=True)
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    model: AlignSam = args.model
    if args.ckpt_path is None:
        print('no checkpoint provided')
    else:
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
    model.to(dtype=torch.bfloat16,  device='cuda')
    # R = np.random.RandomState(42)
    build_phrase_mapping()
    precision = HalfPrecision('bf16-true')
    # data = orjson.loads((PROCESSED_VG_DATA_ROOT / 'CT-RATE/train.json').read_bytes())
    from mmmm.data.dataset.local import get_local_data_list
    data = get_local_data_list('BTCV-Abdomen', Split.TRAIN)
    for item_idx, item in enumerate(tqdm(data)):
        # findings_ann: str = item['findings-ann']
        # findings_remove_ann = findings_ann.replace('<p>', '').replace('</p>', '')
        # if findings_ann != findings_remove_ann:
        #     pass
        # phrases = parse_phrases(findings_ann)
        # supported_phrases = []
        # unsupported_phrases = []
        # for phrase in phrases:
        #     if (target := phrase_mapping.get(normalize(phrase))) is None:
        #         unsupported_phrases.append(phrase)
        #     else:
        #         supported_phrases.append((phrase, target.name))
        # print(supported_phrases)
        # print(unsupported_phrases)
        # targets = list({target for _, target in supported_phrases})
        # if 'adrenal gland' in targets:
        #     targets = ['adrenal gland']
        #     targets.extend(['left adrenal gland', 'right adrenal gland', 'kidney', 'left kidney', 'right kidney'])
        # for image_path in item['image']:
        case_dir = item['dataset_dir'] / 'data' / item['key']
        image_path = case_dir / 'images.pt.zst'
        sparse = Sparse.from_json((case_dir / 'sparse.json').read_bytes())
        targets = [
            target.name
            for target in cytoolz.concat(sparse.targets.values())
        ]
        image = load_pt_zst(image_path)
        image = tvtf.to_dtype(image, torch.float, scale=True)
        roi_size = (32, 256, 256)
        vit_patch_size = (4, 16, 16)
        masks = load_pt_zst(case_dir / 'masks.pt.zst').cuda()
        # masks = nnf.interpolate(masks[None].float(), trans_image.shape[1:], mode='trilinear')[0] > 0.5
        targets_dict = {
            target.name: target
            for y in sparse.targets.values()
            for target in y
        }
        batch = _collate_fn(
            [
                {
                    'patch_size': vit_patch_size,
                    'classes': targets,
                },
            ]
        )
        batch = precision.convert_input(batch)
        image, _ = ensure_rgb(image, contiguous=True)
        image_plot = image.clone()
        # image = intensity_norm(image)
        image = precision.convert_input(image).cuda()
        batch = move_data_to_device(batch, 'cuda')

        def _patch_infer(patch: torch.Tensor):
            batch['image'] = [patch[0]]
            patch_logits = model(batch)[0][None]
            return patch_logits

        with torch.inference_mode():
            for rotate in [0, 1]:
                masks_logits = sliding_window_inference(
                    image[None].rot90(dims=(-1, -2), k=rotate),
                    roi_size,
                    8,
                    _patch_infer,
                    0.8,
                    BlendMode.GAUSSIAN,
                    progress=True,
                )[0]
                sem_masks_pred = masks_logits.sigmoid() > 0.5
                for i, target in enumerate(targets):
                    label = einops.reduce(masks[slice(*targets_dict[target].index_offset)], 'c ... -> ...', 'any')
                    label = label.rot90(dims=(-1, -2), k=rotate)
                    dice = _dice(sem_masks_pred[i], label) * 100
                    print(target, f'rotate={rotate}', f'{dice.item():.2f}')
                    # IndexTrackerBinary(
                    #     image_plot,
                    #     torch.stack([label, sem_masks_pred[i]]),
                    #     # sem_masks_pred[i, 0:1],
                    #     # torch.stack([label[i], sem_masks_pred[i, 0]]),
                    #     choose_max=True,
                    #     title=f'{image_path}\n{target}\ndice={dice:.1f}',
                    #     # title=f'{image_path}\n{target}',
                    # )

if __name__ == '__main__':
    main()
