import json
from pathlib import Path

from lightning.fabric.utilities import move_data_to_device
import nibabel as nib
import numpy as np
import torch
from transformers import GenerationConfig
from transformers.generation import GenerateDecoderOnlyOutput

from luolib.lightning import PeftTrainer
from monai import transforms as mt
from monai.metrics import DiceMetric

from mmmm.data import MMMMDataModule
from mmmm.data.dataset.local import SamplePatch
from mmmm.models import MMMMForCausalLM
from mmmm.tokenizer import MMMMTokenizer

from scripts.cli import CLI

class DataModule(MMMMDataModule):
    def predict_data(self):
        return self.train_data()

    def predict_transform(self):
        conf = self.trans_conf
        return mt.Compose([
            SamplePatch(
                conf.patch_size,
                conf.vit_patch_size,
                conf.num_pos,
                conf.num_neg,
                self.tokenizer,
                inference=True,
            ),
            InputTransformD(),
        ])

    def get_eval_collate_fn(self):
        return self.get_train_collate_fn()

def main():
    cli = CLI(
        model_class=MMMMForCausalLM,
        datamodule_class=DataModule,
        trainer_class=PeftTrainer,
        run=False,
    )
    output_dir = Path('infer-test')
    model = cli.model
    tokenizer: MMMMTokenizer = cli.config_init.tokenizer
    datamodule = cli.datamodule
    ckpt_dir = Path('output/phase-1/base-vl-sam_lr2/seed-42/run-20240317_022436-1q63yj3u/checkpoint/step=40000.ckpt')
    model.load_default_adapter(ckpt_dir)
    model: MMMMForCausalLM = model.cuda()
    model = model.to(dtype=torch.bfloat16)
    for data_idx, data in enumerate(datamodule.predict_dataloader()):
        data = move_data_to_device(data, 'cuda')
        image = data['image'].to(torch.bfloat16)
        data['image'] = image
        # gen_kwargs = {
        #     "max_new_tokens": 1500,
        #     "do_sample": False,
        # }
        gen_config = GenerationConfig(
            max_new_tokens=250,
            do_sample=False,
            # num_beams=8,
        )
        with torch.no_grad():
            outputs: GenerateDecoderOnlyOutput = model.generate(
                generation_config=gen_config, image=image, return_dict_in_generate=True, output_hidden_states=True,
                **data['vlm_inputs'],
            )
        mask_classes = data.pop('mask_classes')[0]
        name_to_idx = {name: i for i, name in enumerate(mask_classes)}
        masks: torch.BoolTensor = data.pop('masks')[0]
        pos_classes, neg_classes = [], []
        is_pos_mask = masks.any(dim=(1, 2, 3))
        for i, name in enumerate(mask_classes):
            if is_pos_mask[i]:
                pos_classes.append(name)
            else:
                neg_classes.append(name)
        print('positive:', pos_classes)
        print('negative:', neg_classes)
        hidden_states = torch.cat([hidden_states[-1] for hidden_states in outputs.hidden_states], dim=1)
        seq = outputs.sequences
        text = tokenizer.decode(seq[seq != tokenizer.unk_token_id])
        print(text)
        pred_targets = tokenizer.parse_targets(seq)[0]
        if pred_targets is None:
            print('parse targets failed')
            continue
        seg_token_mask = tokenizer.create_seg_token_mask(seq[:, 1:])
        seg_hidden_states = [hidden_states[0, seg_token_mask[0]]]
        mask_logits = model._generate_and_postprocess_masks(image, seg_hidden_states)[0]
        mask_pred = mask_logits.sigmoid() > 0.5
        fp = {}
        fn = {}
        mask_pred_mc = mask_pred.new_zeros(mask_pred.shape[1:], dtype=torch.uint8)
        pos_pred_targets = {}
        num_pos_pred = 0
        dice = {}
        for i, name in enumerate(pred_targets):
            pred_pos = mask_pred[i].any()
            if pred_pos:
                num_pos_pred += 1
                mask_pred_mc[mask_pred[i]] = num_pos_pred
                pos_pred_targets[str(num_pos_pred)] = name
            if (class_idx := name_to_idx.get(name)) is not None:
                if is_pos_mask[class_idx]:
                    dice_metric = DiceMetric()
                    dice[name] = (dice_metric(mask_pred[None, i:i + 1], masks[None, class_idx:class_idx + 1]) * 100).item()
                    print(f'{name}: {dice[name]:.1f}')
                    if not pred_pos:
                        fn[name] = masks[class_idx].sum().item()
                elif pred_pos:
                    # let's assume that model will not output names unmentioned
                    fp[name] = mask_pred[i].sum().item()
        print(f'FP: {fp}')
        print(f'FN: {fn}')
        data_dir = output_dir / f'data-{data_idx}'
        data_dir.mkdir(exist_ok=True, parents=True)
        spacing = data['spacing'][0].cpu().numpy()
        affine = np.diag([*spacing, 1])
        nib.save(
            nib.Nifti1Image(image[0, 0].float().cpu().numpy(), affine),
            data_dir / 'patch.nii.gz',
        )
        nib.save(
            nib.Nifti1Image(mask_pred_mc.cpu().numpy(), affine),
            data_dir / 'mask-pred.nii.gz',
        )
        mean_dice = np.mean(list(dice.values())).item()
        print(f'mean positive dice: {mean_dice:.1f}')
        dice['_mean'] = mean_dice
        info = {
            'key': data['key'][0],
            'text': text,
            'FP': fp,
            'FN': fn,
            'names': pos_pred_targets,
            'dice': dice,
        }
        (data_dir / 'info.json').write_text(json.dumps(info, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()
