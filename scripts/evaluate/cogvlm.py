import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer


from luolib.utils import load_pt_zst


def setup_cogvlm(checkpoint: str, tokenizer: str):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer)

    model = model.to('cuda')
    model = model.eval()

    return model, tokenizer


def cogvlm_collate_fn(batch: list[dict]):
    assert len(batch) == 1

    if batch[0]['image'].endswith('.pt.zst'):
        image = load_pt_zst(batch[0]['image'])
    else:
        image = Image.open(batch[0]['image']).convert('RGB')
    return {
        'image': image,
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }


def cogvlm_vl_evaluate(model, tokenizer, dataloader):
    results = []

    for sample in tqdm(dataloader):
        inputs = model.build_conversation_input_ids(
            tokenizer, query=sample['question'], images=[sample['image']]
        )

        with torch.inference_mode():
            prediction = (
                tokenizer.decode(
                    model.generate(
                        input_ids=inputs['input_ids'].unsqueeze(0).to('cuda'),
                        token_type_ids=inputs['token_type_ids'].unsqueeze(0).to('cuda'),
                        attention_mask=inputs['attention_mask'].unsqueeze(0).to('cuda'),
                        images=[[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
                        max_new_tokens=256,
                    )[0],
                    skip_special_tokens=True,
                )
                .split('Answer: ')[1]
                .strip()
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