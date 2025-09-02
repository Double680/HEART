# HybTQA


### Dataset Preparation

Download [MultihierTT](https://github.com/psunlpgroup/MultiHiertt) datasets and put the `train/dev/test.json` files into `./datasets/multihiertt`.

### Python Environment

Prepare basic conda environment:
```
conda create -n hybtqa python=3.12
conda activate hybtqa
pip install torch==2.7.1
pip install -r requirements.txt
```

Download flash-attn from [release](https://github.com/Dao-AILab/flash-attention/releases) and install it locally:
```
pip install flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```
