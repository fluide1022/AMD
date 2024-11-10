# AMD: Autoregressive Motion Diffusion
![visualization](./assets/single02.png)
![visualization](./assets/single01.png)



<p align="center">
  <a href='https://arxiv.org/abs/2305.09381'>
    <img src='https://img.shields.io/badge/Arxiv-2312.06553-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://www.youtube.com/watch?v=5A60JwzwWXM'>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a>
  </a>
</p>


<p align="center">
<!-- <h1 align="center">InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion</h1> -->
<strong>AMD: Autoregressive Motion Diffusion</strong></h1>
   <p align="center">
    <a href='https://scholar.google.com/citations?user=5XsDL6kAAAAJ&hl=zh-CN' target='_blank'>Bo Han</a>&emsp;
    <a href='' target='_blank'>Hao Peng</a>&emsp;
    <a href='https://www.cs.cityu.edu.hk/~minjdong/' target='_blank'>Minjing Dong</a>&emsp;
    <a href='https://rayeren.github.io/' target='_blank'>Yi Ren</a>&emsp;
    <a href='' target='_blank'>Yixuan Shen</a>&emsp;
    <a href='http://changxu.xyz/' target='_blank'>Chang Xu</a>&emsp;
    <br>
    Zhejiang University &emsp; Unity China &emsp;
    University of Sydney &emsp; National University of Singapore
    <br>
  </p>
</p>




## 📜 TODO List

- [ ] Release the HumanLong3D Dataset
- [ ] Release the HumanMusic Dataset
- [x] Release the main codes for implementation.

## ⚙️ Content

```
📁 Code/
├── data_loaders/                      # Data loading related
│   ├── humanml/
│   │   ├── data/
│   │   │   └── dataset.py
│   │   └── utils/
│   │       └── plot_script.py        # Skeleton visualization
│   ├── get_data.py
│   └── tensors.py
│
├── diffusion/                         # Core diffusion model
│   └── gaussian_diffusion.py
│
├── eval/                              # Evaluation scripts
│   ├── eval.py                       # Single motion evaluation
│   ├── eval_AUTOREG.py              # Compound motion evaluation
│   └── eval_T2L.py                  # Motion duration prediction network evaluation
│
├── model/                             # Model definitions
│   ├── amd_autoreg.py               # Compound motion model
│   └── amd.py                       # Single motion model
│
├── train/                             # Training scripts
│   ├── train_amd_single.py          # Single motion training
│   └── train_amd_autoreg.py         # Compound motion training
│
├── utils/                             # Utility functions
│   ├── parser_util.py               # Parse running parameters
│   └── model_util.py                # Parse model parameters
│
├── visualize/                         # Visualization tools
│   └── joints2smpl                   # Skeleton to SMPL conversion
│
├── text2length.py                     # Motion duration prediction
├── 0_amd_single_generate.py          # Single motion generation
└── 1_amd_autoreg_generate.py         # Compound motion generation
```

## 🏃 Training

### Single Motion Model
```bash
nohup python -m train.train_amd_single \
    --save_dir save/0_humanml3d_single \
    --data_dir ./dataset/HumanLong3D \
    --device 1 \
    --overwrite \
    > ./save/0_humanlong3d_single/train.log 2>&1 &
```

### Compound Motion Model
```bash
nohup python -m train.train_amd_autoreg \
    --save_dir save/0_humanlong3d_autoreg \
    --data_dir ./dataset/HumanLong3D \
    --device 0 \
    --overwrite \
    > ./save/0_humanlong_autoreg/train.log 2>&1 &
```

### Motion Duration Predictor
```bash
nohup python train_length_est.py \
    --name t2l \
    --gpu_id 2 \
    --dataset_name t2m \
    > ./checkpoints/t2m/train.log 2>&1 &
```

## 📊 Evaluation

### Single Motion Evaluation
#### Without Duration Prediction
```bash
nohup python -m eval.eval \
    --model_path '' \
    --eval_mode mm_short \
    --device 0 \
    > ./save/xxx/0_eval_mm.log 2>&1 &
```

#### With Duration Prediction
```bash
nohup python -m eval.eval_T2L \
    --model_path '' \
    --eval_mode mm_short \
    --device 1 \
    > ./save/xxx/0_eval_mm_T2L.log 2>&1 &
```

### Compound Motion Evaluation
```bash
nohup python -m eval.eval_AUTOREG \
    --model_path '' \
    --eval_mode mm_short \
    --device 3 \
    > ./save/xxx/0_eval_mm_AUTOREG.log 2>&1 &
```

### Duration Predictor Evaluation
```bash
python eval_length_est.py \
    --name t2l \
    --gpu_id 0 \
    --dataset_name t2m
```

## 🎮 Synthesis

### Single Motion Generation
```bash
python 0_amd_single_generate.py \
    --model_path 'path/to/your/model' \
    --text "a person is walking" \
    --device 0
```

### Compound Motion Generation 
```bash
python 1_amd_autoreg_generate.py \
    --model_path 'path/to/your/model' \
    --text "a person walks forward then jumps" \
    --device 0
```

## 🤝 Citation

If you find this repository useful for your work, please consider citing it as follows:

```
@article{Han2024,
  title={AMD: Autoregressive Motion Diffusion},
  author={Bo Han, Hao Peng, Minjing Dong, Yi Ren, Yixuan Shen, Chang Xu},
  journal={AAAI},
  year={2024}
}
```
