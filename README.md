<div align="center">

## LoRaSeek: Boosting Denoising Ability in Neural-enhanced LoRa Decoder via Hierarchical Feature Extraction

**ACM MobiCom 2025**

Khang Nguyen¹, Yidong Ren¹, Jialuo Du¹, Jingkai Lin¹  
Maolin Gan¹, Shigang Chen², Mi Zhang³, Chunyi Peng⁴, Zhichao Cao¹  

¹ Michigan State University · ² University of Florida · ³ The Ohio State University · ⁴ Purdue University

[![Paper](https://img.shields.io/badge/Paper-ACM%20-blue)](https://dl.acm.org/doi/10.1145/3680207.3765241)
[![Project Page](https://img.shields.io/badge/Project-Page%20-red)](https://dl.acm.org/doi/10.1145/3680207.3765241)
![Conference](https://img.shields.io/badge/ACM-MobiCom%202025-orange)

</div>

---

## Overview

- A **hierarchical U-Net** with CNN and hybrid Transformer for multi-scale signal representation
- A **hybrid, lightweight Transformer** with channel scaling and local-enhanced FFN.
- **Dual attention-based skip connections** for preserving important chirp characteristics across scales
- Decoding using 2 options: **LoRaPHY vs. Light-weight DNN**
---


## Getting Started

### Installation

```bash
cd loraseek

conda create -n loraseek python=3.10
conda activate loraseek

pip install -r requirements.txt
```

---

## Dataset
- Dataset: [LoRaSeek HF Dataset](https://huggingface.co/datasets/kangnguyen/LoRaSeek-Dataset) 
- Pretrained checkpoints: [LoRaSeek HF Model](https://huggingface.co/kangnguyen/LoRaSeek_SF7_Large)
---

## Training

`main.py` trains the model for `--epochs` epochs and then evaluates it on the held-out split. The dataset directory is expected to contain `.mat` files named
`{label}_{snr}_{sf}_{bw}_{batch}_{label}_{packet}_{symbol}.mat`, with a clean of each symbol at SNR `35` used as the ground truth.

```bash
python main.py \
    --data_dir /path/to/sf7_dataset \
    --sf 7 \
    --epochs 50 \
    --batch_size 64 
```

Useful options:

| Option | Default | Description |
| --- | --- | --- |
| `--snr_list` | `-35 ... -11` | SNR levels (dB) to load from the dataset |
| `--train_ratio` | `0.8` | Fraction of samples used for training; the rest is used for testing |
| `--alpha` / `--beta` |  | Weights of the spectrogram MSE loss and the classification loss |
| `--use_checkpoint` | off | Initialize from `--ckpt_path` before training |

Checkpoints (`epoch{N}_model_{model_name}.pth`), sample spectrograms and `log_file.txt` are written to `output/SF{sf}-{timestamp}/`.

Note: You can customize your model by adjusting architectural settings such as channel size, hidden dimensions, number of layers, and other configuration parameters

---

## Evaluation

Set `--epochs 0` to skip training and only evaluate a checkpoint:

```bash
python main.py \
    --data_dir /path/to/sf7_dataset \
    --sf 7 \
    --epochs 0 \
    --use_checkpoint \
    --ckpt_path /path/to/sf7_ckpt
```

Evaluation reports, per SNR level, the symbol accuracy of
- the DNN classifier head (`error_matrix_dnn`), and
- LoRaPHY decoding of the denoised signal (`error_matrix_loraphy`), compared against LoRaPHY on the raw noisy signal as a baseline.

The per-SNR accuracies are printed and saved as `error_matrix_dnn.npy` and `error_matrix_loraphy.npy` in the output directory.

---

## Code
- [X] Models
- [ ] Pretrained checkpoints
- [X] Dataset processing
- [X] Training 
- [X] Evaluation



---



## Citation

```bibtex
@inproceedings{nguyen2025loraseek,
  title={LoRaSeek: Boosting denoising ability in neural-enhanced LoRa decoder via hierarchical feature extraction},
  author={Nguyen, Khang and Ren, Yidong and Du, Jialuo and Lin, Jingkai and Gan, Maolin and Chen, Shigang and Zhang, Mi and Peng, Chunyi and Cao, Zhichao},
  booktitle={Proceedings of the 31st Annual International Conference on Mobile Computing and Networking},
  pages={712--726},
  year={2025}
}
```

---


