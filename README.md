# MMMM

## Setup

```zsh
git clone --recursive https://github.com/function2-llx/MMMM.git
cd MMMM
mamba env create -f environment.yaml
BUILD_MONAI=1 pip install -e third-party/LuoLib/third-party/MONAI
mamba activate mmmm
echo "export PYTHONPATH=$PWD:$PWD/third-party/LuoLib" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
