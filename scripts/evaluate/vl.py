from einops import repeat
import json
from jsonargparse import CLI
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT
from utils import (
    setup_seed,
    setup_radfm,
    radfm_collate_fn,
    setup_medflamingo,
    medflamingo_collate_fn,
    NLPMetrics,
)


class VQATestDataset(Dataset):
    def __init__(self, dataset: str):
        super().__init__()
        self.name = dataset
        with open(PROCESSED_VL_DATA_ROOT / dataset / "test.json") as f:
            self.dataset = [
                {"image": x["image"][0], **vqa}
                for x in json.load(f)
                for vqa in x["vqa"]
            ]

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class VLEvaluator:
    def __init__(
        self,
        checkpoint: str,
        tokenizer: str,
        task: str,
        dataset: str,
        seed: int = 233,
        num_workers: int = 8,
        output_dir: str = "results/",
    ):
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.task = task
        if task == 'vqa':
            self.dataset = VQATestDataset(dataset)
        self.num_workers = num_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = NLPMetrics()
        setup_seed(seed)

    def radfm(self):
        model, tokenizer = setup_radfm(self.checkpoint, self.tokenizer)

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=radfm_collate_fn,
        )
        results = []

        for sample in tqdm(dataloader):
            language = tokenizer(
                sample["question"],
                max_length=2048,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )["input_ids"].to("cuda")
            vision = sample["image"].to("cuda")
            prediction = tokenizer.decode(
                model.generate(language, vision)[0], skip_special_tokens=True
            )

            results.append(
                {
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prediction": prediction,
                    **self.metrics.compute(prediction, sample["answer"]),
                },
            )

            print(sample["question"])
            print(sample["answer"])
            print(prediction)

        results = pd.DataFrame(results)
        results.to_csv(self.output_dir / f"{self.task}_{self.dataset.name}_radfm_zeroshot.csv")
        with open(self.output_dir / f"{self.task}_{self.dataset.name}_radfm_zeroshot.json", "w") as f:
            json.dump(
                {
                    "bleu": results["bleu"].mean(),
                    "rouge": results["rouge"].mean(),
                    "meteor": results["meteor"].mean(),
                    "bertscore": results["bertscore"].mean(),
                    "exact_match": results["exact_match"].mean(),
                },
                f,
                indent=4,
            )

    def medflamingo(self):
        model, processor = setup_medflamingo(self.checkpoint, self.tokenizer)

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=medflamingo_collate_fn,
        )
        results = []

        for sample in tqdm(dataloader):
            language = processor.encode_text(sample["question"])
            vision = processor.preprocess_images([sample["image"]])
            vision = repeat(vision, "1 c h w -> 1 1 1 c h w")
            prediction = (
                processor.tokenizer.decode(
                    model.generate(
                        vision_x=vision.to("cuda"),
                        lang_x=language["input_ids"].to("cuda"),
                        attention_mask=language["attention_mask"].to("cuda"),
                        max_new_tokens=256,
                    )[0]
                )
                .replace("<unk> ", "")
                .strip()
            )

            results.append(
                {
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prediction": prediction,
                    **self.metrics.compute(prediction, sample["answer"]),
                },
            )

            print(sample["question"])
            print(sample["answer"])
            print(prediction)

        results = pd.DataFrame(results)
        results.to_csv(self.output_dir / f"{self.task}_{self.dataset.name}_medflamingo_zeroshot.csv")
        with open(self.output_dir / f"{self.task}_{self.dataset.name}_medflamingo_zeroshot.json", "w") as f:
            json.dump(
                {
                    "bleu": results["bleu"].mean(),
                    "rouge": results["rouge"].mean(),
                    "meteor": results["meteor"].mean(),
                    "bertscore": results["bertscore"].mean(),
                    "exact_match": results["exact_match"].mean(),
                },
                f,
                indent=4,
            )


if __name__ == "__main__":
    CLI(VLEvaluator, as_positional=False)
