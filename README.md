# MMMM

## Setup

```zsh
git clone --recursive https://github.com/function2-llx/MMMM.git
cd MMMM
mamba env create -f environment.yaml
BUILD_MONAI=1 pip install --no-build-isolation -e third-party/LuoLib/third-party/MONAI
mamba activate mmmm
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo \
"export PYTHONPATH=$PWD:$PYTHONPATH
export BUILD_MONAI=1" \
>> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

## Data Preparation

Download the datasets (MIMIC-CXR, CT-RATE, etc.) and extract them to `data/origin/<data type>/<dataset name>`, where `<data type>` can be `local` for image datasets with localization (bounding boxes, segmentation) annotations and `vision-language` for VQA and radiology report datasets.

Then execute pre-processing scripts for each dataset. For instance, for MIMIC-CXR, execute the script at `scripts/data/vl/MIMIC-CXR/MIMIC-CXR.py` to pre-process the data. After the pre-processing is finished, the pre-processed data are placed at `data/processed/vision-language/MIMIC-CXR`, where `<split>.json` specifies the data items for each split.

## Training

[THUDM/cogvlm-chat-hf](https://huggingface.co/THUDM/cogvlm-chat-hf) is used as the base VLM. 

The example commands for running the three-stage training of VividMed are as follows. Please adapt your number of devices and batch size accordingly. Note that we disable `torch.compile` due the compatibility issue with our dependencies, enable it if you're ready to address it.

```zsh
# Stage 1: Visual Grounding Pre-training
python scripts/cli.py fit -c conf/phase-vg/fit.yaml --compile false --data.dataloader.train_batch_size ... --trainer.accumulate_grad_batches ... --seed_everything $RANDOM --model.freeze_sam false --model.freeze_isam false
# Stage 2: Medical Visual Instruction Tuning
python scripts/cli.py fit -c conf/phase-vlm/fit.yaml --compile false --data.dataloader.train_batch_size ... --trainer.accumulate_grad_batches ... --seed_everything $RANDOM
# Stage 3: Alignment (grounded report generate)
python scripts/cli.py fit -c conf/phase-grg/fit.yaml --compile false --data.dataloader.train_batch_size ... --trainer.accumulate_grad_batches ... --seed_everything $RANDOM --model.freeze_sam false --model.freeze_isam false
```
