# MMMM

## Setup

```zsh
git clone --recursive https://github.com/function2-llx/MMMM.git
cd MMMM
mamba env create -f environment.yaml
BUILD_MONAI=1 pip install --no-build-isolation -e third-party/LuoLib/third-party/MONAI
mamba activate mmmm
echo \
"export PYTHONPATH=$PWD:$PWD/third-party/LuoLib
export BUILD_MONAI=1" \
>> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

```
, and build xformers from source / install pre-built wheel.

Manually modify `pad_nd` according to: https://github.com/Project-MONAI/MONAI/issues/7842
