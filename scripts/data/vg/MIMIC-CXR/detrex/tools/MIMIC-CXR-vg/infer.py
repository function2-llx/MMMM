from functools import partial
from pathlib import Path

import orjson
import torch
import torchvision.transforms.v2.functional as tvtf
from itk.support.extras import image
from tqdm import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch, create_ddp_model
from detectron2.structures import Instances
from detrex.data.datasets.register_vindr_cxr import thing_classes
from projects.dino_eva.modeling import DINO

local_labels = {
    'Aortic enlargement': 'aortic enlargement',
    'Atelectasis': 'atelectasis',
    'Calcification': 'calcification',
    'Cardiomegaly': 'cardiomegaly',
    'Clavicle fracture': 'clavicle fracture',
    'Consolidation': 'pulmonary consolidation',
    'Edema': 'pulmonary edema',
    'Emphysema': 'pulmonary emphysema',
    'Enlarged PA': 'pulmonary artery enlargement',
    'ILD': 'interstitial lung disease',
    'Infiltration': 'pulmonary infiltrate',
    'Lung cavity': 'pulmonary cavity',
    'Lung cyst': 'pulmonary cyst',
    'Lung Opacity': 'pulmonary opacification',
    'Mediastinal shift': 'mediastinal shift',
    'Nodule/Mass': 'lung nodule',
    # 'Other lesion': 'other lesion',
    'Pleural effusion': 'pleural effusion',
    'Pleural thickening': 'pleural thickening',
    'Pneumothorax': 'pneumothorax',
    'Pulmonary fibrosis': 'pulmonary fibrosis',
    'Rib fracture': 'rib fracture',
}

thing_classes = [local_labels.get(target) for target in thing_classes]
assert len(thing_classes) == len(set(thing_classes))
thing_class_to_idx = {
    target: i
    for i, target in enumerate(thing_classes)
}

def _dataset_func(split: str):
    data = orjson.loads(Path(f'data/processed/visual-grounding/MIMIC-CXR/{split}.json').read_bytes())
    ret = []
    for item in tqdm(data):
        for image_path, plane in zip(item['image'], item['plane']):
            if plane not in ('PA', 'AP'):
                continue
            image_path = Path(image_path)
            image_path = Path(
                'data/origin/vision-language/MIMIC-CXR-JPG/files',
                *image_path.parts[-4:-1],
                image_path.name.replace('.pt.zst', '.jpg'),
            )
            class_names = sorted(set(target for tag in item['tags'] if (target := tag['target']) in thing_class_to_idx))
            ret.append({
                'file_name': image_path,
                'class_names': class_names,
                'classes': torch.tensor([thing_class_to_idx[name] for name in class_names]),
                'report': item['filtered_tagged_findings'],
            })

    return ret

def _register():
    for split in ['train', 'test']:
        name = f'mimic-cxr_vg_{split}'
        DatasetCatalog.register(name, partial(_dataset_func, split))
        meta = MetadataCatalog.get(name)
        meta.thing_classes = thing_classes

_register()

score_th = 0.1

def select_instances(classes: torch.Tensor, instances: Instances):
    ret = {}
    score_mask = instances.scores >= score_th
    for class_idx in classes.tolist():
        class_mask = instances.pred_classes == class_idx
        select_mask = score_mask & class_mask
        if not select_mask.any() and class_mask.any():
            first_index = class_mask.byte().argmax()
            select_mask[first_index] = True
        if select_mask.any():
            ret[thing_classes[class_idx]] = instances[select_mask].pred_boxes.tensor.tolist()
    return ret

def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg.dataloader.test.dataset.names = ['mimic-cxr_vg_train', 'mimic-cxr_vg_test']
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    model: DINO = instantiate(cfg.model)
    model.to(cfg.train.device).eval()
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    dataloader = instantiate(cfg.dataloader.test)
    with torch.inference_mode():
        for inputs in tqdm(dataloader):
            for input_item in inputs:
                input_item['image'] = tvtf.equalize(input_item['image'])
            outputs = model(inputs)
            for input_item, output_item in zip(inputs, outputs):
                results = select_instances(input_item['classes'], output_item['instances'])
                origin_path: Path = input_item['file_name']
                image_save_path = Path('data/processed/visual-grounding/MIMIC-CXR/image', *origin_path.parts[-4:])
                image_save_path.parent.mkdir(exist_ok=True, parents=True)
                if image_save_path.exists():
                    image_save_path.unlink()
                image_save_path.hardlink_to(origin_path)
                image_save_path.with_name(f'{image_save_path.stem}_box.json').write_bytes(
                    orjson.dumps(results, option=orjson.OPT_INDENT_2),
                )
    model.eval()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

