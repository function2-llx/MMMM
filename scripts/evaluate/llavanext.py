from PIL import Image
import sys


def setup_llavanext(checkpoint: str, tokenizer: str):
    sys.path.append('third-party/LLaVA')

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=checkpoint,
        model_base=None,
        model_name=get_model_name_from_path(checkpoint),
    )

    return tokenizer, model, image_processor, context_len

def llavanext_collate_fn(batch):
    assert len(batch) == 1

    return {
        'image': Image.open(batch[0]['image']).convert('RGB'),
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }