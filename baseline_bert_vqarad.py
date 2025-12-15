import os
import json
from collections import Counter
import re
import string

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
MAX_ANSWERS = 300
MAX_Q_LEN = 32
OUTPUT_DIR = "./vqa_baseline_bert"



def normalize_answer(s: str) -> str:

    def lower(text):
        return text.lower()

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_single(pred: str, gold: str) -> float:

    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)



def load_vqa_rad():
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    return dataset


def build_answer_vocab(train_split, max_answers=MAX_ANSWERS):

    all_answers = [ex["answer"] for ex in train_split]
    counter = Counter(all_answers)

    sorted_answers = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if max_answers is not None:
        sorted_answers = sorted_answers[:max_answers]

    answer_list = [a for a, _ in sorted_answers]
    answer_list = ["<unk>"] + answer_list

    answer2id = {a: i for i, a in enumerate(answer_list)}
    id2answer = {i: a for a, i in answer2id.items()}

    return answer2id, id2answer, counter



IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class VQARADDataset(Dataset):
    def __init__(self, hf_split, answer2id, transform=None):

        self.data = hf_split
        self.answer2id = answer2id
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        image = ex["image"]  # PIL Image
        if self.transform is not None:
            image = self.transform(image)

        question = ex["question"]
        answer = ex["answer"]

        if answer in self.answer2id:
            label = self.answer2id[answer]
        else:
            label = self.answer2id["<unk>"]

        raw_ans_type = ex["answer_type"] if "answer_type" in ex else ""
        if raw_ans_type is None:
            raw_ans_type = ""
        ans_type = str(raw_ans_type)

        sample = {
            "image": image,          # Tensor
            "question": question,    # str
            "answer": answer,        # str
            "label": label,          # int
            "answer_type": ans_type
        }
        return sample




class VQABertBaseline(nn.Module):
    def __init__(self, num_classes, freeze_bert=True):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.img_feat_dim = resnet.fc.in_features  # 2048

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.text_feat_dim = self.bert.config.hidden_size

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        fusion_dim = 512
        self.fusion = nn.Linear(self.img_feat_dim + self.text_feat_dim, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, images, questions):

        img_feat = self.cnn_backbone(images)         # (B, 2048, 1, 1)
        img_feat = torch.flatten(img_feat, 1)        # (B, 2048)

        enc = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=MAX_Q_LEN,
            return_tensors="pt"
        ).to(images.device)

        bert_out = self.bert(**enc)
        last_hidden = bert_out.last_hidden_state     # (B, seq_len, hidden_size)
        cls_emb = last_hidden[:, 0, :]              # (B, hidden_size) 对应 [CLS]

        fused = torch.cat([img_feat, cls_emb], dim=1)  # (B, 2048 + 768)
        fused = torch.relu(self.fusion(fused))         # (B, fusion_dim)
        logits = self.classifier(fused)                # (B, num_classes)
        return logits



def evaluate(model, data_loader, id2answer):
    model.eval()
    total = 0
    correct_cls = 0
    em_sum = 0.0
    f1_sum = 0.0

    closed_n = closed_em = closed_f1 = 0.0
    open_n = open_em = open_f1 = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Eval"):
            images = batch["image"].to(DEVICE)
            questions = batch["question"]         # list[str]
            labels = batch["label"].to(DEVICE)    # (B,)
            answers = batch["answer"]             # list[str]
            ans_types = batch["answer_type"]      # list[str]

            logits = model(images, questions)
            preds = torch.argmax(logits, dim=1)   # (B,)

            total += labels.size(0)
            correct_cls += (preds == labels).sum().item()

            for i in range(len(answers)):
                pred_id = preds[i].item()
                pred_str = id2answer[pred_id]
                gt_str = answers[i]

                em = 1.0 if normalize_answer(pred_str) == normalize_answer(gt_str) else 0.0
                f1 = f1_score_single(pred_str, gt_str)

                em_sum += em
                f1_sum += f1

                ans_type = ans_types[i]
                if ans_type:
                    at_lower = ans_type.lower()
                    if "close" in at_lower:   # "CLOSED"
                        closed_n += 1
                        closed_em += em
                        closed_f1 += f1
                    elif "open" in at_lower: # "OPEN"
                        open_n += 1
                        open_em += em
                        open_f1 += f1

    acc = correct_cls / total if total > 0 else 0.0
    em_all = em_sum / total if total > 0 else 0.0
    f1_all = f1_sum / total if total > 0 else 0.0

    print("\n====== Baseline-BERT  ======")
    print(f"Classification Accuracy: {acc:.4f}")
    print(f"All     - EM: {em_all:.4f}, Token F1: {f1_all:.4f}")

    if closed_n > 0:
        print(f"Closed  - EM: {closed_em/closed_n:.4f}, Token F1: {closed_f1/closed_n:.4f}")
    if open_n > 0:
        print(f"Open    - EM: {open_em/open_n:.4f}, Token F1: {open_f1/open_n:.4f}")

    return {
        "accuracy": acc,
        "all": (em_all, f1_all),
        "closed": (closed_em / closed_n if closed_n > 0 else None,
                   closed_f1 / closed_n if closed_n > 0 else None),
        "open": (open_em / open_n if open_n > 0 else None,
                 open_f1 / open_n if open_n > 0 else None),
    }



def train_baseline_bert():
    raw_dataset = load_vqa_rad()
    train_raw = raw_dataset["train"]
    test_raw = raw_dataset["test"]

    answer2id, id2answer, counter = build_answer_vocab(train_raw, max_answers=MAX_ANSWERS)
    num_classes = len(answer2id)

    train_ds = VQARADDataset(train_raw, answer2id, transform=train_transform)
    test_ds = VQARADDataset(test_raw, answer2id, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = VQABertBaseline(num_classes=num_classes, freeze_bert=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}"):
            images = batch["image"].to(DEVICE)
            questions = batch["question"]
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(images, questions)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} avg_loss: {avg_loss:.4f}")

        print("在测试集上评估（Accuracy + EM/F1 + closed/open）...")
        evaluate(model, test_loader, id2answer)

    # 保存模型和答案映射
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "vqa_baseline_bert.pt")
    torch.save(model.state_dict(), model_path)
    mapping_path = os.path.join(OUTPUT_DIR, "answer2id.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(answer2id, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    train_baseline_bert()
