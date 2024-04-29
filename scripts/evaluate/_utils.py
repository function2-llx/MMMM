import numpy as np
from PIL import Image
import random
import torch
from torchvision import transforms


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def radfm_collate_fn(batch: list[dict]):
    assert len(batch) == 1
    if batch[0]["image"].endswith(".pt"):
        image = torch.load(batch[0]["image"])
        image = (image - image.min()) / (image.max() - image.min())
    else:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    [512, 512],
                    scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )
        image = transform(Image.open(batch[0]["image"]).convert("RGB"))
    target_d, max_d = 4, 4
    if len(image.shape) == 4:
        max_d = max(image.shape[3], max_d)
    for temp_d in range(4, 65, 4):
        if abs(temp_d - max_d) < abs(target_d - max_d):
            target_d = temp_d
    if len(image.shape) == 3:
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0).unsqueeze(-1), size=(512, 512, target_d)
        ).unsqueeze(0)
    else:
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(512, 512, target_d)
        ).unsqueeze(0)
    question_list = [False for _ in range(len(str(batch[0]["question"])))]
    question = ""
    if random.random() < 0.5:
        position = 0
    else:
        position = len(question_list) - 1
    question_list[position] = True
    for i in range(len(question_list)):
        if question_list[i]:
            question += (
                "<image>"
                + "".join([f"<image{i}>" for i in range(32)])
                + "</image>"
                + batch[0]["question"][i]
            )
        else:
            question += batch[0]["question"][i]
    return {
        "image": image,
        "question": question,
        "answer": batch[0]["answer"],
    }
