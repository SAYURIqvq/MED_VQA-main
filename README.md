# MED_VQA — Reproducing VQA-RAD Experiments (BERT+CNN Baselines, BLIP Fine-tuning, Grad-CAM)

This repo contains training / evaluation scripts for **medical visual question answering** on the **VQA-RAD** dataset from Hugging Face (`flaviagiammarino/vqa-rad`).

It includes:
1) **Baseline-BERT VQA classifier** (ResNet50 + BERT, BERT frozen)
2) **Fine-tuned BERT** (freeze CNN, train BERT + head with different LRs)
3) **Partial fine-tune BERT** (unfreeze last N BERT layers, default N=4)
4) **Grad-CAM** visualizations for the BERT baseline (loads a trained checkpoint and produces heatmaps)
5) **BLIP fine-tuning** on VQA-RAD (sequence generation model)
6) **Comparison script**: compares a (separate) LSTM baseline model vs your fine-tuned BLIP model on test set

---

## 0) What you need (super explicit)

### Hardware
- **BERT+CNN training**: GPU recommended (scripts auto-use CUDA if available).
- **BLIP training scripts are CPU-only by default** (very slow).  
  If you want GPU BLIP training, change:
  ```python
  DEVICE = torch.device("cpu")
  ```
  to:
  ```python
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

### Data
No manual download needed. The scripts call:
```python
load_dataset("flaviagiammarino/vqa-rad")
```
which auto-downloads and caches the dataset.

---

## 1) Environment setup (pinned versions)

> Repro tip: Use **Python 3.10** to avoid many dependency edge cases.

### Option A (Conda, recommended)
Create `environment.yml`:
```yaml
name: med_vqa
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10.13
  - pip=23.3.2
  - numpy=1.26.4
  - matplotlib=3.8.3
  - pillow=10.2.0
  - tqdm=4.66.2
  - pytorch=2.1.2
  - torchvision=0.16.2
  - pytorch-cuda=11.8
  - pip:
      - transformers==4.37.2
      - datasets==2.18.0
      - accelerate==0.27.2
      - huggingface-hub==0.20.3
```

Then:
```bash
conda env create -f environment.yml
conda activate med_vqa
```

### Option B (Pip + existing PyTorch)
Install PyTorch however you normally do (CPU/GPU), then:
```bash
pip install transformers==4.37.2 datasets==2.18.0 accelerate==0.27.2 tqdm==4.66.2 pillow==10.2.0 matplotlib==3.8.3 numpy==1.26.4
```

---

## 2) Repository layout (important)

Recommended layout:
```
MED_VQA/
  baseline_bert_vqarad.py
  baseline_bert_finetune_vqarad.py
  baseline_bert_partial_finetune_vqarad.py
  baseline_bert_gradcam_vqarad.py
  compare.py
  blip/
    blip_version.py
    blip_vqarad_longtrain.py
```

This matters because `compare.py` expects the BLIP fine-tuned model at:
`./blip/blip_medvqa_vqarad`

So you should run BLIP training **from inside the `blip/` folder** (see below).

---

## 3) Experiments: Baseline-BERT (ResNet50 + BERT)

### 3.1 Baseline-BERT (BERT frozen)
Script: `baseline_bert_vqarad.py`

Default settings (from code):
- `BATCH_SIZE=8`, `EPOCHS=10`, `LR=1e-4`
- `MAX_ANSWERS=300`, `MAX_Q_LEN=32`
- Training augmentation: random flip + random rotation(10°)
- Builds answer vocab from the **top 300 most frequent answers**, plus `<unk>` for everything else
- Evaluates: **classification accuracy + EM + token F1**, and also open/closed breakdown
- Saves to `./vqa_baseline_bert/`

Run (from repo root):
```bash
python baseline_bert_vqarad.py
```

Expected outputs:
- `./vqa_baseline_bert/vqa_baseline_bert.pt`
- `./vqa_baseline_bert/answer2id.json`

---

### 3.2 Fine-tune BERT (freeze CNN, train BERT + head)
Script: `baseline_bert_finetune_vqarad.py`

Default settings (from code):
- `BATCH_SIZE=4`, `EPOCHS=8`
- `LR_BERT=5e-6`, `LR_HEAD=1e-4`
- CNN frozen, BERT trainable
- Two learning rates (BERT params vs head params)
- Gradient clipping `max_norm=1.0`
- Saves to `./vqa_baseline_bert_finetune/`

Run:
```bash
python baseline_bert_finetune_vqarad.py
```

Expected outputs:
- `./vqa_baseline_bert_finetune/vqa_baseline_bert_finetune.pt`
- `./vqa_baseline_bert_finetune/answer2id.json`

---

### 3.3 Partial fine-tune BERT (unfreeze last 4 layers)
Script: `baseline_bert_partial_finetune_vqarad.py`

Default settings (from code):
- `BATCH_SIZE=4`, `EPOCHS=10`
- `LR_BERT=5e-6`, `LR_HEAD=1e-4`
- Unfreezes last `unfreeze_last_n_layers=4` of BERT encoder (and pooler)
- Saves to `./vqa_baseline_bert_partial_finetune/`

Run:
```bash
python baseline_bert_partial_finetune_vqarad.py
```

Expected outputs:
- `./vqa_baseline_bert_partial_finetune/vqa_baseline_bert_partial_finetune.pt`
- `./vqa_baseline_bert_partial_finetune/answer2id.json`

---

## 4) Grad-CAM Visualization (Baseline-BERT)

Script: `baseline_bert_gradcam_vqarad.py`

Default behavior:
- Runs on CPU by default
- Loads this checkpoint by default:
  - `./vqa_baseline_bert_finetune/vqa_baseline_bert_finetune.pt`
  - `./vqa_baseline_bert_finetune/answer2id.json`
- Saves heatmaps into: `./gradcam_baseline_bert/`
- Loads weights with `strict=False` and prints missing/unexpected keys (expected)

Run (after you have the finetune outputs):
```bash
python baseline_bert_gradcam_vqarad.py
```

---

## 5) BLIP fine-tuning on VQA-RAD

You have two BLIP scripts:
- `blip/blip_version.py` saves to `./blip_medvqa_vqarad`
- `blip/blip_vqarad_longtrain.py` saves to `./blip_medvqa_longtrain`

Both scripts:
- use `MODEL_NAME = "Salesforce/blip-vqa-base"`
- default: `BATCH_SIZE=2`, `EPOCHS=6`, `MAX_Q_LEN=32`, `MAX_A_LEN=8`
- evaluate “exact match” accuracy on the first N samples each epoch
- save Hugging Face format via `model.save_pretrained()` and `processor.save_pretrained()`

### 5.1 Train BLIP (default)
Run from repo root:
```bash
cd blip
python blip_version.py
cd ..
```

This will create:
```
blip/blip_medvqa_vqarad/
  config.json
  generation_config.json
  model.safetensors
  preprocessor_config.json
  tokenizer_config.json
  vocab.txt
  ...
```

That matches what `compare.py` expects (`./blip/blip_medvqa_vqarad`).

### 5.2 Train BLIP (longtrain variant)
```bash
cd blip
python blip_vqarad_longtrain.py
cd ..
```

---

## 6) Compare BLIP vs a separate LSTM baseline model

Script: `compare.py`

What it does:
- Loads VQA-RAD test set and evaluates:
  - a baseline **ResNet18 + LSTM** classifier (not the BERT baseline),
  - and your **fine-tuned BLIP** model.

### 6.1 Required files for the LSTM baseline
`compare.py` requires these files in the repo root:
- `baseline_word2idx.json`
- `baseline_answer2idx.json`
- `baseline_model_best.pth`

**If you do NOT have `baseline_model_best.pth`, `compare.py` will crash.**  
In that case, either:
- (A) add the file, or
- (B) comment out the baseline part and only evaluate BLIP.

### 6.2 Run
From repo root:
```bash
python compare.py
```

---

## 7) Reproducibility notes (so others get similar results)

### Randomness / seeds
- BERT baselines use random image augmentation (flip/rotation), so results may vary run-to-run.
- BLIP scripts call `torch.manual_seed(42)` inside training.

If you want “as close as possible” reproducibility across machines, add this near the top of each training script:
```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 8) Common issues (fast fixes)

### “BLIP is extremely slow”
BLIP scripts are CPU-only by default. Switch device to CUDA (see section 0).

### “Grad-CAM prints missing/unexpected keys”
Expected: the script loads weights with `strict=False`.

### “GitHub push fails because files are too big”
Add this to `.gitignore`:
```gitignore
*.pth
*.pt
*.safetensors
```
and remove large files from git history if needed (or use Git LFS).

---

## 9) Quick “copy-paste” reproduction checklist

From repo root:

```bash
# 1) create env (see section 1)
conda activate med_vqa

# 2) Baseline BERT (frozen)
python baseline_bert_vqarad.py

# 3) Fine-tune BERT
python baseline_bert_finetune_vqarad.py

# 4) Partial fine-tune BERT
python baseline_bert_partial_finetune_vqarad.py

# 5) Grad-CAM (after fine-tune)
python baseline_bert_gradcam_vqarad.py

# 6) BLIP fine-tune
cd blip
python blip_version.py
cd ..

# 7) Compare (needs baseline_model_best.pth etc)
python compare.py
```
