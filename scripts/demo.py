from pathlib import Path

from jsonargparse import ArgumentParser
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.plugins import HalfPrecision
import orjson
from transformers import GenerationConfig
from transformers.generation import GenerateDecoderOnlyOutput

from mmmm.data.datamodule import _collate_fn
from mmmm.data.defs import ConvTurn, PROCESSED_VL_DATA_ROOT
from mmmm.data.utils import prepare_vlm_inputs
from mmmm.misc import image_transform

base_pool_size_z = 2
pool_size_xy = 2
max_tokens_z = 4
base_patch_size_z = 16
patch_size_xy = 16
max_vision_tokens = 100

def main():
    parser = ArgumentParser()
    parser.add_argument('adapter_dir', type=Path)
    args = parser.parse_args()
    from mmmm.models.mmmm import from_pretrained
    model, tokenizer = from_pretrained('conf/model.yaml', args.adapter_dir)
    model = model.cuda()
    data: list = orjson.loads(Path(PROCESSED_VL_DATA_ROOT / 'VQA-RAD/test.json').read_bytes())
    precision = HalfPrecision('bf16-true')
    for sample in data:
        conv = []
        image, patch_size, pool_size, num_vision_tokens = image_transform(sample['image'][0])
        for i in range(len(sample['vqa'])):
            prompt = sample['vqa'][i]['question']
            conv.append(ConvTurn(prompt, ''))
            vlm_inputs, _ = prepare_vlm_inputs(
                conv,
                tokenizer,
                num_vision_tokens,
                inference=True,
                grounding=False,
            )
            input_len = len(vlm_inputs['input_ids'])

            batch = _collate_fn([
                {
                    'image': image,
                    'vlm_inputs': vlm_inputs,
                    'patch_size': patch_size,
                    'pool_size': pool_size,
                }
            ])
            batch = precision.convert_input(batch)
            batch = move_data_to_device(batch, 'cuda')
            gen_config = GenerationConfig(max_new_tokens=512, do_sample=False)
            output: GenerateDecoderOnlyOutput = model.generate(
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_hidden_states=True,
                **batch['vlm_inputs'],
                image=batch['image'],
                patch_size=batch['patch_size'],
                pool_size=batch['pool_size'],
            )
            token_ids = output.sequences[0].tolist()
            token_ids = token_ids[input_len:]
            if token_ids[-1] == tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
            response = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
            print(prompt)
            print(response)
            conv[-1] = ConvTurn(prompt, response)

if __name__ == '__main__':
    main()
