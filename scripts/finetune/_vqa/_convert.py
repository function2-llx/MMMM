import torch
from PIL import Image
from einops import rearrange, repeat
from peft import PeftModel
from torchvision import transforms
from transformers import LlamaTokenizer

from luolib.utils.zstd import load_pt_zst


ckpt_path = "/data/MMMM/RadFM/pytorch_model.bin"
tokenizer_path = "/data/MMMM/RadFM"

adapter_path = "/home/chenxuanzhong/MMMM/output/finetune/VQA-RAD/radfm_vqa-rad/seed-42/run-20240620_202638-csnd1hdx/checkpoint/last.ckpt/adapter"

from RadFM.multimodality_model import MultiLLaMAForCausalLM

model = MultiLLaMAForCausalLM(
    lang_model_path=tokenizer_path,
)
checkpoint = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint)

inference_model = PeftModel.from_pretrained(model, adapter_path)

inference_model = inference_model.to('cuda')
inference_model.eval()

print(1)

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
special_tokens = {
    'additional_special_tokens': [f'<image{i}>' for i in range(32)] + ['<image>', '</image>']
}
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2

image_path = "data/processed/vision-language/VQA-RAD/images/synpic100176.jpg"

query = "Is there evidence of a pneumoperitoneum?"


if image_path.endswith('.pt'):
    image = rearrange(torch.load(image_path).float(), 'c d h w -> c h w d')
    image = (image - image.min()) / (image.max() - image.min())
elif image_path.endswith('.pt.zst'):
    image = rearrange(load_pt_zst(image_path).float(), 'c d h w -> c h w d')
    image = (image - image.min()) / (image.max() - image.min())
else:
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    image = transform(image)

target_d, max_d = 4, 4
if len(image.shape) == 4:
    max_d = max(image.shape[3], max_d)
for temp_d in range(4, 65, 4):
    if abs(temp_d - max_d) < abs(target_d - max_d):
        target_d = temp_d
if len(image.shape) == 3:
    image = torch.nn.functional.interpolate(
        repeat(image, 'c h w -> 1 c h w 1'), size=(512, 512, target_d)
    ).unsqueeze(0)
else:
    if image.shape[0] == 1:
        image = torch.nn.functional.interpolate(
            repeat(image, '1 h w d -> 1 3 h w d'), size=(512, 512, target_d)
        ).unsqueeze(0)
    else:
        image = torch.nn.functional.interpolate(
            repeat(image, 'c h w d -> 1 c h w d'), size=(512, 512, target_d)
        ).unsqueeze(0)

image = image.to('cuda')
prompt = '<image>' + ''.join([f'<image{i}>' for i in range(32)]) + '</image>' + query
language = tokenizer(
    prompt,
    return_tensors='pt',
)['input_ids']

generation = model.generate(language, image)
generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
print(generated_texts[0])

# def ft_iblip():
#     from instructblip import FinetuneInstructBlip
#
#     dataset = "Slake"
#     run_name = "run-20240518_210121-b2q9jsmg"
#
#     ckpt_path = f"/home/chenxuanzhong/MMMM/output/finetune/{dataset}/instructblip-13b/seed-42/{run_name}/checkpoint/last.ckpt"
#     ckpt = torch.load(ckpt_path)
#     state_dict = ckpt['state_dict']
#
#
#     model = FinetuneInstructBlip(model_path="/data/new_llm/instructblip-vicuna-13b")
#
#     incompatible_keys = model.load_state_dict(state_dict, strict=False)
#     print("Missing keys:", incompatible_keys.missing_keys)
#     print("Unexpected keys:", incompatible_keys.unexpected_keys)
#
#
#     save_model = model.instructblip_model
#
#     for name, param in save_model.named_parameters():
#         print(name, param.size())
#     print(type(state_dict))
#     save_path = f"/data/MMMM/output/finetune/{dataset}/instructblip-13b/seed-42/{run_name}/"
#     #
#     save_model.save_pretrained(save_path)
#
#
#     print("done")
#
#
# def test_cogvlm():
#     import torch
#     from peft import PeftModel
#     from transformers import AutoModelForCausalLM, LlamaTokenizer
#     import torch.nn.functional as nnf
#     import einops
#     import torch.nn as nn
#
#     from torchvision.io import read_image, ImageReadMode
#     from torchvision.transforms.v2 import functional as tvtf
#     from luolib.types import tuple3_t
#
#     dataset = "Slake"
#     run_name = "run-20240522_231522-gt3d5ubb"
#
#     adapter_path = f"/home/chenxuanzhong/MMMM/output/finetune/{dataset}/cogvlm-chat-hf_slake/seed-42/{run_name}/checkpoint/last.ckpt/adapter"
#
#     tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
#     model = AutoModelForCausalLM.from_pretrained(
#         'THUDM/cogvlm-chat-hf',
#         torch_dtype=torch.bfloat16,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True
#     ).to('cuda').eval()
#
#     pos_embed = model.model.vision.patch_embedding.position_embedding.weight
#     cls_pos_embed, pos_embed = pos_embed[0:1], pos_embed[1:]
#     pos_embed = einops.rearrange(pos_embed, '(h w) c -> 1 c h w', h=35, w=35)
#
#     pos_embed = nnf.interpolate(pos_embed, (16, 16), mode='area')
#     pos_embed = torch.cat([cls_pos_embed, einops.rearrange(pos_embed, '1 c h w ->(h w) c')])
#     model.model.vision.patch_embedding.position_embedding = nn.Embedding(
#         *pos_embed.shape[:2], _weight=pos_embed,
#     )
#
#
#     inference_model = PeftModel.from_pretrained(model, adapter_path)
#
#     image_path = "/data/MMMM/data/processed/vision-language/Slake/images/xmlab1/source.jpg"
#     query = "What modality is used to take this image?"
#
#     def build_conversation_input_ids(tokenizer, image_path, query):
#         def intensity_norm_(
#             image: torch.Tensor,
#             mean: tuple3_t[float] = (0.48145466, 0.4578275, 0.40821073),
#             std: tuple3_t[float] = (0.26862954, 0.26130258, 0.27577711),
#         ):
#             """default mean and std is adopted from CogVLM (, which is from CLIP)"""
#             mean = image.new_tensor(mean)
#             std = image.new_tensor(std)
#             x = image.view(image.shape[0], -1)
#             x.sub_(mean[:, None]).div_(std[:, None])
#
#         image = read_image(image_path, ImageReadMode.RGB)
#         image = tvtf.to_dtype(image, torch.float32, scale=True)
#         image = tvtf.resize(image, (224, 224))
#         intensity_norm_(image)
#
#         prompt = f'Question: {query} Answer:'
#         prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
#
#         input_ids = torch.cat([
#             torch.tensor([tokenizer.bos_token_id]).unsqueeze(1),
#             prompt_ids.unsqueeze(1),
#             torch.tensor([tokenizer.eos_token_id]).unsqueeze(1),
#         ])
#
#         input_ids = input_ids.reshape(input_ids.shape[-1], input_ids.shape[0])
#
#         LANGUAGE_TOKEN_TYPE = 0
#         VISION_TOKEN_TYPE = 1
#         num_vision_tokens = 16 * 16 + 2
#         seq_len = input_ids.shape[1]
#         token_type_ids = torch.full(
#             (input_ids.shape[0], num_vision_tokens + seq_len), LANGUAGE_TOKEN_TYPE,
#         )
#         token_type_ids[:, 1:1 + num_vision_tokens] = VISION_TOKEN_TYPE
#
#         attention_mask = torch.ones_like(input_ids)
#
#         new_input_ids = torch.zeros_like(token_type_ids)
#         new_input_ids[token_type_ids == LANGUAGE_TOKEN_TYPE] = input_ids.view(-1)
#         new_attn_mask = torch.ones_like(token_type_ids)
#         new_attn_mask[token_type_ids == LANGUAGE_TOKEN_TYPE] = attention_mask.view(-1)
#
#         inputs = {
#             'input_ids': new_input_ids.to('cuda'),
#             'token_type_ids': token_type_ids.to('cuda'),
#             'attention_mask': new_attn_mask.to('cuda'),
#             'images': [[image.to('cuda').to(torch.bfloat16)]]
#         }
#         return inputs
#
#     inputs = build_conversation_input_ids(tokenizer, image_path, query)
#
#     gen_kwargs = {"max_length": 2048, "do_sample": False}
#
#     with torch.no_grad():
#         outputs = inference_model.generate(**inputs, **gen_kwargs)
#         outputs = outputs[:, inputs['input_ids'].shape[1] + 1: -1]
#         print(tokenizer.decode(outputs[0]))
#
# def llava_next():
#     from transformers import LlavaNextProcessor, AutoTokenizer
#     import torch
#     from torchvision.io import read_image, ImageReadMode
#     from torchvision.transforms.v2 import functional as tvtf
#     from luolib.types import tuple3_t
#     from peft import PeftModel
#     import einops
#     from _vqa.llavanext import MyLlavaNextForConditionalGeneration
#
#     def build_conversation_input_ids(tokenizer, image_path, query):
#         def intensity_norm_(
#             image: torch.Tensor,
#             mean: tuple3_t[float] = (0.48145466, 0.4578275, 0.40821073),
#             std: tuple3_t[float] = (0.26862954, 0.26130258, 0.27577711),
#         ):
#             """default mean and std is adopted from CogVLM (, which is from CLIP)"""
#             mean = image.new_tensor(mean)
#             std = image.new_tensor(std)
#             x = image.view(image.shape[0], -1)
#             x.sub_(mean[:, None]).div_(std[:, None])
#
#         image = read_image(image_path, ImageReadMode.RGB)
#         image = tvtf.to_dtype(image, torch.float32, scale=True)
#         image = tvtf.resize(image, (224, 224))
#         image_size = torch.tensor((224, 224))
#         intensity_norm_(image)
#
#         prompt = f'<image>\nQuestion: {query} Answer:'
#         prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
#         input_ids = torch.cat([
#             torch.tensor([tokenizer.bos_token_id]).unsqueeze(1),
#             prompt_ids.unsqueeze(1),
#         ])
#
#         input_ids = input_ids.reshape(input_ids.shape[-1], input_ids.shape[0])
#         inputs = {
#             'input_ids': input_ids.to('cuda'),
#             'pixel_values': einops.repeat(image.to('cuda').unsqueeze(0), 'n ... -> n l2 ...', l2=2),
#             'image_sizes': image_size.to('cuda').unsqueeze(0),
#             'attention_mask': torch.ones_like(input_ids).to('cuda'),
#         }
#         return inputs
#
#     adapter_path = "/data/MMMM/output/finetune/VQA-RAD/llava-next/seed-42/run-20240531_213429-fcmma2zq/checkpoint/last.ckpt/adapter"
#     model_path = "llava-hf/llava-v1.6-vicuna-13b-hf"
#
#     processor = LlavaNextProcessor.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = MyLlavaNextForConditionalGeneration.from_pretrained(
#         model_path,
#         image_grid_pinpoints=[[224, 224]],
#         vision_config={
#             "hidden_size": 1024,
#             "image_size": 224,
#             "intermediate_size": 4096,
#             "model_type": "clip_vision_model",
#             "num_attention_heads": 16,
#             "num_hidden_layers": 24,
#             "patch_size": 14,
#             "projection_dim": 768,
#             "vocab_size": 32000
#           },
#     )
#     model.to('cuda')
#
#     inference_model = PeftModel.from_pretrained(model, adapter_path)
#     inference_model.to('cuda')
#
#
#     image_path = "/data/MMMM/data/processed/vision-language/VQA-RAD/images/synpic19118.jpg"
#     query = "Are these small opacities in the right lung calcifications?"
#     inputs = build_conversation_input_ids(tokenizer, image_path, query)
#
#
#     # autoregressively complete prompt
#     output = inference_model.generate(**inputs, max_new_tokens=100)
#     prefix_len = len(f"\nQuestion: {query} Answer:")
#     print(processor.decode(output[0], skip_special_tokens=True)[prefix_len + 1:])
#
# def llava_med():
#     import torch
#     import einops
#     import torch.nn as nn
#     from llava.model.builder import load_pretrained_model
#     from llava.mm_utils import get_model_name_from_path
#     from llava.constants import DEFAULT_IMAGE_TOKEN
#     from torchvision.io import read_image, ImageReadMode
#     from torchvision.transforms.v2 import functional as tvtf
#     from luolib.types import tuple3_t
#     from peft import PeftModel
#
#     model_id = "microsoft/llava-med-v1.5-mistral-7b"
#     model_name = get_model_name_from_path(model_id)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(model_id, None, model_name)
#
#     pos_embed = model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight
#     cls_pos_embed, pos_embed = pos_embed[0:1], pos_embed[1:]
#     pos_embed = einops.rearrange(pos_embed, '(h w) c -> 1 c h w', h=24, w=24)
#     import torch.nn.functional as nnf
#     pos_embed = nnf.interpolate(pos_embed, (16, 16), mode='area')
#     pos_embed = torch.cat([cls_pos_embed, einops.rearrange(pos_embed, '1 c h w ->(h w) c')])
#     model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(
#         *pos_embed.shape[:2], _weight=pos_embed,
#     )
#     model.model.vision_tower.vision_tower.vision_model.embeddings.position_ids = torch.arange(257).expand((1, -1))
#
#     adapter_path = "/data/MMMM/output/finetune/VQA-RAD/llava-med1.5/seed-42/run-20240608_204409-wobl572z/checkpoint/last.ckpt/adapter"
#     inference_model = PeftModel.from_pretrained(model, adapter_path)
#     inference_model.to('cuda')
#
#     query = "Are these small opacities in the right lung calcifications?"
#
#     # conv = conv_templates["vicuna_v1"].copy()
#     # conv.append_message(conv.roles[0], prompt)
#     # conv.append_message(conv.roles[1], None)
#     # prompt = conv.get_prompt()
#
#     def build_conversation_input_ids(tokenizer, image_path, query):
#         def intensity_norm_(
#                 image: torch.Tensor,
#                 mean: tuple3_t[float] = (0.48145466, 0.4578275, 0.40821073),
#                 std: tuple3_t[float] = (0.26862954, 0.26130258, 0.27577711),
#         ):
#             """default mean and std is adopted from CogVLM (, which is from CLIP)"""
#             mean = image.new_tensor(mean)
#             std = image.new_tensor(std)
#             x = image.view(image.shape[0], -1)
#             x.sub_(mean[:, None]).div_(std[:, None])
#
#         image = read_image(image_path, ImageReadMode.RGB)
#         image = tvtf.to_dtype(image, torch.float32, scale=True)
#         image = tvtf.resize(image, (224, 224))
#         intensity_norm_(image)
#
#         prompt = f'{DEFAULT_IMAGE_TOKEN}\nQuestion: {query} Answer:'
#         prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
#
#         input_ids = torch.cat([
#             torch.tensor([tokenizer.bos_token_id]).unsqueeze(1),
#             prompt_ids.unsqueeze(1),
#         ])
#
#         input_ids = input_ids.reshape(input_ids.shape[-1], input_ids.shape[0])
#
#         inputs = {
#             'input_ids': input_ids.to('cuda'),
#             'attention_mask': torch.ones_like(input_ids).to('cuda'),
#             'image': image.to('cuda').to(torch.bfloat16)
#         }
#         return inputs
#
#     image_path = "/data/MMMM/data/processed/vision-language/VQA-RAD/images/synpic19118.jpg"
#
#     inputs = build_conversation_input_ids(tokenizer, image_path, query)
#
#     with torch.inference_mode():
#         output_ids = inference_model.generate(
#             inputs["input_ids"],
#             images=inputs["image"].unsqueeze(0).half().cuda(),
#             do_sample=False,
#             max_new_tokens=1024,
#             use_cache=True)
#
#     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#
#     print(outputs)
#
#
# def m3d():
#     # Prepare your 3D medical image:
#     # 1. The image shape needs to be processed as 1*32*256*256, consider resize and other methods.
#     # 2. The image needs to be normalized to 0-1, consider Min-Max Normalization.
#     # 3. The image format needs to be converted to .npy
#     # 4. Although we did not train on 2D images, in theory, the 2D image can be interpolated to the shape of 1*32*256*256 for input.
#
#     import torch
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#
#     from torchvision.io import read_image, ImageReadMode
#     from torchvision.transforms.v2 import functional as tvtf
#
#     device = torch.device('cuda') # 'cpu', 'cuda'
#     dtype = torch.bfloat16 # or bfloat16, float16, float32
#
#     model_name_or_path = '/data/new_llm/M3D-LaMed-Llama-2-7B'
#     proj_out_num = 256
#
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name_or_path,
#         torch_dtype=dtype,
#         device_map='auto',
#         trust_remote_code=True)
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name_or_path,
#     )
#
#     # OpenI
#     def build_RG_inputs_ids(tokenizer, image_path, prompt):
#         image_tokens = "<im_patch>" * proj_out_num
#         prompt = image_tokens + prompt
#
#         prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
#         input_ids = torch.cat([
#             torch.tensor([tokenizer.bos_token_id]).unsqueeze(1),
#             prompt_ids.unsqueeze(1),
#         ])
#
#         input_ids = input_ids.reshape(input_ids.shape[-1], input_ids.shape[0])
#         input_ids = input_ids.to(device=device)
#
#         image = read_image(image_path, ImageReadMode.GRAY)
#         image = tvtf.to_dtype(image, torch.float32, scale=True)
#         image = tvtf.resize(image, (256, 256))
#
#         image = einops.repeat(image, '1 h w -> 1 1 d h w', d=32)
#         image_pt = image.to(dtype=dtype, device=device)
#
#         inputs = {
#             'input_ids': input_ids,
#             'attention_mask': torch.ones_like(input_ids).to('cuda'),
#             'image': image_pt
#         }
#         return inputs
#
#     adapter_path = "/home/chenxuanzhong/MMMM/output/finetune/OpenI/m3d_openi/seed-42/run-20240614_232340-9t7twlpk/checkpoint/last.ckpt/adapter"
#     image_path = "data/processed/vision-language/OpenI/images/2788_IM-1222-1001.dcm.png"
#     # Report Generation
#     prompt = f'Please write a radiology report for me:'
#     inputs = build_RG_inputs_ids(tokenizer, image_path, prompt)
#
#     #VQA
#     def build_VQA_input_ids(tokenizer, image_path, query):
#         image = read_image(image_path, ImageReadMode.GRAY)
#         image = tvtf.to_dtype(image, torch.float32, scale=True)
#         image = tvtf.resize(image, (256, 256))
#         image = einops.repeat(image, '1 h w -> 1 1 d h w', d=32)
#         image = image.to(dtype=dtype)
#
#         prompt = f'Question: {query} Answer:'
#         prompt = "<im_patch>" * proj_out_num + prompt
#         prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
#
#         input_ids = torch.cat([
#             torch.tensor([tokenizer.bos_token_id]).unsqueeze(1),
#             prompt_ids.unsqueeze(1),
#         ])
#
#         input_ids = input_ids.reshape(input_ids.shape[-1], input_ids.shape[0])
#
#         inputs = {
#             'input_ids': input_ids.to('cuda'),
#             'attention_mask': torch.ones_like(input_ids).to('cuda'),
#             'image': image.to('cuda').to(torch.bfloat16)
#         }
#         return inputs
#
#
#     adapter_path = "/data/MMMM/output/finetune/VQA-RAD/m3d/seed-42/run-20240614_232324-9jt8s3y8/checkpoint/last.ckpt/adapter"
#
#     image_path = "data/processed/vision-language/VQA-RAD/images/synpic100176.jpg"
#     query = "Is there evidence of a pneumoperitoneum?"
#
#     inputs = build_VQA_input_ids(tokenizer, image_path, query)
#
#
#     inference_model = PeftModel.from_pretrained(model, adapter_path)
#     inference_model = inference_model.to(device=device)
#
#
#     generation = inference_model.generate(inputs['image'], inputs['input_ids'], max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=1.0)
#     generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
#     print(generated_texts[0])
#
