# DuGI-MAE: Improving Infrared Mask Autoencoders via Dual-Domain Guidance
## Pretrain

### Create a conda environment and activate it:
```bash
conda create -n infmae python=3.7
conda activate DUGImae
```

### Install Pytorch==1.8.0 and torchvision==0.9.0 with CUDA==11.1:
```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

### Install timm==0.3.2:
```bash
pip install timm==0.3.2
```

### Pretrain on Inf-590K
```bash
python main_pretrain_DuGI_MAE.py --world_size 1 --local_rank 0 --rank 0
```

## Dataset
### Inf-590K dataset: 
https://pan.baidu.com/s/1831SVvBQnrFuy-bg9esPPg    1id2
