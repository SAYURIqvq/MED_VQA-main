# MED_VQA医学视觉问答系统（Medical VQA）

基于 **VQA-RAD 数据集** 的医学视觉问答（Medical Visual Question Answering）项目，系统性实现并对比多种深度学习方法（BERT+CNN、不同微调策略、BLIP），并结合 **Grad-CAM 可解释性分析**，支持可复现实验与多模态 AI 研究。

<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/c6c3fb46-f5a4-4bcd-a23f-35a24c31e9d7" />


---

## 项目特点

* 多模型对比：BERT+CNN、全量微调、部分微调、BLIP
* 多模态建模：图像（CNN）+ 文本（BERT）
* 可解释性：Grad-CAM 可视化模型关注区域
* 完整实验流程：训练 / 评估 / 对比 / 可视化
* 可复现：固定依赖版本 + 标准数据加载方式

---

## 项目结构

```text
MED_VQA-main/
├── blip/                                            # BLIP 相关代码、训练产物与示例图像
│   ├── blip_version.py                              # BLIP 标准训练脚本
│   ├── blip_vqarad_longtrain.py                     # BLIP 长训练版本脚本
│   ├── example_test_image_blip.jpg                  # BLIP 示例测试图像
│   ├── blip_medvqa_vqarad/                          # BLIP 标准训练输出目录
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── preprocessor_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── blip_medvqa_longtrain/                       # BLIP 长训练输出目录
│
├── gradcam_baseline_bert/                           # Grad-CAM 可视化结果
│   ├── gradcam_baseline_idx0.jpg
│   ├── gradcam_baseline_idx1.jpg
│   ├── gradcam_baseline_idx2.jpg
│   ├── gradcam_baseline_idx3.jpg
│   └── gradcam_baseline_idx4.jpg
│
├── vqa_baseline_bert/                               # BERT+CNN 基线输出目录
│   └── answer2id.json
│
├── vqa_baseline_bert_finetune/                      # BERT 全量微调输出目录
│   └── answer2id.json
│
├── vqa_baseline_bert_partial_finetune/              # BERT 部分微调输出目录
│   └── answer2id.json
│
├── baseline_bert_vqarad.py                          # BERT+CNN 基线训练脚本
├── baseline_bert_finetune_vqarad.py                 # BERT 全量微调训练脚本
├── baseline_bert_partial_finetune_vqarad.py         # BERT 部分微调训练脚本
├── baseline_bert_gradcam_vqarad.py                  # Grad-CAM 可解释性分析脚本
├── compare.py                                       # 模型对比脚本
├── baseline_word2idx.json                           # baseline 词表映射
├── baseline_answer2idx.json                         # baseline 答案映射
└── README.md

```
---

## 实验方法

### 1️⃣ Baseline：BERT + CNN（ResNet50）

* 图像编码：ResNet50
* 文本编码：BERT（冻结）
* 输出：分类（Top-K 答案）
* 数据增强：随机翻转 + 旋转

---

### 2️⃣ BERT 微调策略

#### ✅ 全量微调

* 冻结 CNN，仅训练 BERT + 分类头
* 双学习率策略（BERT vs Head）

#### ✅ 部分微调

* 仅解冻 BERT 最后 N 层（默认 4 层）
* 平衡训练效率与性能

---

### 3️⃣ BLIP 模型（生成式 VQA）

* 模型：`Salesforce/blip-vqa-base`
* 任务：生成式问答（Seq2Seq）
* 评估指标：Exact Match

---

### 4️⃣ Grad-CAM 可解释性

* 对 BERT+CNN 模型生成热力图
* 可视化模型在图像中的关注区域
* 支持医学场景解释性分析

---

## 数据集

* 数据来源：Hugging Face
  `flaviagiammarino/vqa-rad`
* 示例图：
  <img width="1024" height="1291" alt="image" src="https://github.com/user-attachments/assets/5a8d92b8-99ff-4789-8f1d-13859f471494" />


---

## 环境配置

### 推荐环境

* Python 3.10
* PyTorch 2.x
* Transformers 4.37+

### 安装依赖（pip）

```bash
pip install transformers==4.37.2 \
            datasets==2.18.0 \
            accelerate==0.27.2 \
            tqdm==4.66.2 \
            pillow==10.2.0 \
            matplotlib==3.8.3 \
            numpy==1.26.4
```

---

## 实验流程

```bash
# 1️⃣ Baseline BERT
python baseline_bert_vqarad.py

# 2️⃣ 微调 BERT
python baseline_bert_finetune_vqarad.py

# 3️⃣ 部分微调
python baseline_bert_partial_finetune_vqarad.py

# 4️⃣ Grad-CAM 可解释性
python baseline_bert_gradcam_vqarad.py

# 5️⃣ BLIP 训练
cd blip
python blip_version.py
cd ..

# 6️⃣ 模型对比
python compare.py
```

---

## 评估指标

* Accuracy（分类准确率）
* Exact Match（精确匹配）
* Token-level F1
* Open / Closed 问题分类评估

---

## License

MIT License
