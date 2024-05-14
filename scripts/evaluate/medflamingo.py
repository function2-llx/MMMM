from PIL import Image
from einops import repeat
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from luolib.utils import load_pt_zst


def setup_medflamingo(checkpoint: str, tokenizer: str):
    from accelerate import Accelerator
    from huggingface_hub import hf_hub_download
    from open_flamingo import create_model_and_transforms

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path='ViT-L-14',
        clip_vision_encoder_pretrained='openai',
        lang_encoder_path=checkpoint,
        tokenizer_path=tokenizer,
        cross_attn_every_n_layers=4,
    )
    medflamingo = hf_hub_download('med-flamingo/med-flamingo', 'model.pt')
    model.load_state_dict(torch.load(medflamingo, map_location='cpu'), strict=False)

    processor = FlamingoProcessor(tokenizer, image_processor)

    accelerator = Accelerator()
    model = accelerator.prepare(model)
    model.eval()

    return model, processor


def medflamingo_collate_fn(batch: list[dict]):
    assert len(batch) == 1

    if batch[0]['image'].endswith('.pt.zst'):
        transform = transforms.ToPILImage()
        image = load_pt_zst(batch[0]['image'])
        image = transform(image.squeeze(1))
    else:
        image = Image.open(batch[0]['image'])
    return {
        'image': image,
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }


class FlamingoProcessor:
    '''
    Processor class for Flamingo.
    '''

    def __init__(self, tokenizer, vision_processor):
        '''
        OF does not use same vision processor, image_processor only transforms single image
        '''
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor

    def encode_text(self, prompt):
        self.tokenizer.padding_side = 'left'
        # For generation padding tokens should be on the left
        return self.tokenizer(
            [prompt],
            return_tensors='pt',
        )

    def preprocess_images(self, images: list):
        vision_x = [self.vision_processor(im).unsqueeze(0) for im in images]
        vision_x = torch.cat(vision_x, dim=0)
        return vision_x


def medflamingo_vl_evaluate(model, processor, dataloader):
    results = []

    for sample in tqdm(dataloader):
        language = processor.encode_text(
            'You are a helpful medical assistant. You are being provided with images and a question about the image, please answer the question. <image>Question:'
            + sample['question']
            + ' Answer:'
        )
        vision = processor.preprocess_images([sample['image']])
        vision = repeat(vision, '1 c h w -> 1 1 1 c h w')
        
        with torch.inference_mode():
            prediction = (
                processor.tokenizer.decode(
                    model.generate(
                        vision_x=vision.to('cuda'),
                        lang_x=language['input_ids'].to('cuda'),
                        attention_mask=language['attention_mask'].to('cuda'),
                        max_new_tokens=512,
                    )[0]
                )
                .replace('<unk> ', '')
                .split('Answer:')[1]
                .strip()
                .split('\n')[0]
            )

        results.append(
            {
                'question': sample['question'],
                'answer': sample['answer'],
                'prediction': prediction,
            },
        )

        print(sample['question'])
        print(sample['answer'])
        print(prediction)

    return results