import torch

def ft_iblip():
    from instructblip import FinetuneInstructBlip

    dataset = "Slake"
    run_name = "run-20240518_210121-b2q9jsmg"

    ckpt_path = f"/home/chenxuanzhong/MMMM/output/finetune/{dataset}/instructblip-13b/seed-42/{run_name}/checkpoint/last.ckpt"
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['state_dict']


    model = FinetuneInstructBlip(model_path="/data/new_llm/instructblip-vicuna-13b")

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", incompatible_keys.missing_keys)
    print("Unexpected keys:", incompatible_keys.unexpected_keys)


    save_model = model.instructblip_model

    for name, param in save_model.named_parameters():
        print(name, param.size())
    print(type(state_dict))
    save_path = f"/data/MMMM/output/finetune/{dataset}/instructblip-13b/seed-42/{run_name}/"
    #
    save_model.save_pretrained(save_path)


    print("done")


def test_cogvlm():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, LlamaTokenizer
    import torch.nn.functional as nnf
    import einops
    import torch.nn as nn

    from torchvision.io import read_image, ImageReadMode
    from torchvision.transforms.v2 import functional as tvtf
    from luolib.types import tuple3_t

    dataset = "Slake"
    run_name = "run-20240522_231522-gt3d5ubb"

    adapter_path = f"/home/chenxuanzhong/MMMM/output/finetune/{dataset}/cogvlm-chat-hf_slake/seed-42/{run_name}/checkpoint/last.ckpt/adapter"

    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to('cuda').eval()

    pos_embed = model.model.vision.patch_embedding.position_embedding.weight
    cls_pos_embed, pos_embed = pos_embed[0:1], pos_embed[1:]
    pos_embed = einops.rearrange(pos_embed, '(h w) c -> 1 c h w', h=35, w=35)

    pos_embed = nnf.interpolate(pos_embed, (16, 16), mode='area')
    pos_embed = torch.cat([cls_pos_embed, einops.rearrange(pos_embed, '1 c h w ->(h w) c')])
    model.model.vision.patch_embedding.position_embedding = nn.Embedding(
        *pos_embed.shape[:2], _weight=pos_embed,
    )


    inference_model = PeftModel.from_pretrained(model, adapter_path)

    image_path = "/data/MMMM/data/processed/vision-language/Slake/images/xmlab1/source.jpg"
    query = "What modality is used to take this image?"

    def build_conversation_input_ids(tokenizer, image_path, query):
        def intensity_norm_(
            image: torch.Tensor,
            mean: tuple3_t[float] = (0.48145466, 0.4578275, 0.40821073),
            std: tuple3_t[float] = (0.26862954, 0.26130258, 0.27577711),
        ):
            """default mean and std is adopted from CogVLM (, which is from CLIP)"""
            mean = image.new_tensor(mean)
            std = image.new_tensor(std)
            x = image.view(image.shape[0], -1)
            x.sub_(mean[:, None]).div_(std[:, None])

        image = read_image(image_path, ImageReadMode.RGB)
        image = tvtf.to_dtype(image, torch.float32, scale=True)
        image = tvtf.resize(image, (224, 224))
        intensity_norm_(image)

        prompt = f'Question: {query} Answer:'
        prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))

        input_ids = torch.cat([
            torch.tensor([tokenizer.bos_token_id]).unsqueeze(1),
            prompt_ids.unsqueeze(1),
            torch.tensor([tokenizer.eos_token_id]).unsqueeze(1),
        ])

        input_ids = input_ids.reshape(input_ids.shape[-1], input_ids.shape[0])

        LANGUAGE_TOKEN_TYPE = 0
        VISION_TOKEN_TYPE = 1
        num_vision_tokens = 16 * 16 + 2
        seq_len = input_ids.shape[1]
        token_type_ids = torch.full(
            (input_ids.shape[0], num_vision_tokens + seq_len), LANGUAGE_TOKEN_TYPE,
        )
        token_type_ids[:, 1:1 + num_vision_tokens] = VISION_TOKEN_TYPE

        attention_mask = torch.ones_like(input_ids)

        new_input_ids = torch.zeros_like(token_type_ids)
        new_input_ids[token_type_ids == LANGUAGE_TOKEN_TYPE] = input_ids.view(-1)
        new_attn_mask = torch.ones_like(token_type_ids)
        new_attn_mask[token_type_ids == LANGUAGE_TOKEN_TYPE] = attention_mask.view(-1)

        inputs = {
            'input_ids': new_input_ids.to('cuda'),
            'token_type_ids': token_type_ids.to('cuda'),
            'attention_mask': new_attn_mask.to('cuda'),
            'images': [[image.to('cuda').to(torch.bfloat16)]]
        }
        return inputs

    inputs = build_conversation_input_ids(tokenizer, image_path, query)

    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        outputs = inference_model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1] + 1: -1]
        print(tokenizer.decode(outputs[0]))

def llava_next():
    from transformers import LlavaNextProcessor, AutoTokenizer
    import torch
    from torchvision.io import read_image, ImageReadMode
    from torchvision.transforms.v2 import functional as tvtf
    from luolib.types import tuple3_t
    from peft import PeftModel
    import einops
    from _vqa.llavanext import MyLlavaNextForConditionalGeneration

    def build_conversation_input_ids(tokenizer, image_path, query):
        def intensity_norm_(
            image: torch.Tensor,
            mean: tuple3_t[float] = (0.48145466, 0.4578275, 0.40821073),
            std: tuple3_t[float] = (0.26862954, 0.26130258, 0.27577711),
        ):
            """default mean and std is adopted from CogVLM (, which is from CLIP)"""
            mean = image.new_tensor(mean)
            std = image.new_tensor(std)
            x = image.view(image.shape[0], -1)
            x.sub_(mean[:, None]).div_(std[:, None])

        image = read_image(image_path, ImageReadMode.RGB)
        image = tvtf.to_dtype(image, torch.float32, scale=True)
        image = tvtf.resize(image, (224, 224))
        image_size = torch.tensor((224, 224))
        intensity_norm_(image)

        prompt = f'<image>\nQuestion: {query} Answer:'
        prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
        input_ids = torch.cat([
            torch.tensor([tokenizer.bos_token_id]).unsqueeze(1),
            prompt_ids.unsqueeze(1),
        ])

        input_ids = input_ids.reshape(input_ids.shape[-1], input_ids.shape[0])
        inputs = {
            'input_ids': input_ids.to('cuda'),
            'pixel_values': einops.repeat(image.to('cuda').unsqueeze(0), 'n ... -> n l2 ...', l2=2),
            'image_sizes': image_size.to('cuda').unsqueeze(0),
            'attention_mask': torch.ones_like(input_ids).to('cuda'),
        }
        return inputs

    adapter_path = "/data/MMMM/output/finetune/VQA-RAD/llava-next/seed-42/run-20240531_213429-fcmma2zq/checkpoint/last.ckpt/adapter"
    model_path = "llava-hf/llava-v1.6-vicuna-13b-hf"

    processor = LlavaNextProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MyLlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        image_grid_pinpoints=[[224, 224]],
        vision_config={
            "hidden_size": 1024,
            "image_size": 224,
            "intermediate_size": 4096,
            "model_type": "clip_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "patch_size": 14,
            "projection_dim": 768,
            "vocab_size": 32000
          },
    )
    model.to('cuda')

    inference_model = PeftModel.from_pretrained(model, adapter_path)
    inference_model.to('cuda')


    image_path = "/data/MMMM/data/processed/vision-language/VQA-RAD/images/synpic19118.jpg"
    query = "Are these small opacities in the right lung calcifications?"
    inputs = build_conversation_input_ids(tokenizer, image_path, query)


    # autoregressively complete prompt
    output = inference_model.generate(**inputs, max_new_tokens=100)
    prefix_len = len(f"\nQuestion: {query} Answer:")
    print(processor.decode(output[0], skip_special_tokens=True)[prefix_len + 1:])


from transformers import AutoTokenizer
from llava import LlavaLlamaForCausalLM
device = "cuda" # the device to load the model onto

model = LlavaLlamaForCausalLM.from_pretrained("microsoft/llava-med-v1.5-mistral-7b")
tokenizer = AutoTokenizer.from_pretrained("microsoft/llava-med-v1.5-mistral-7b")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
