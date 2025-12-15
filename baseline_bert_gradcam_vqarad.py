import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")
BERT_NAME = "bert-base-uncased"

MODEL_PATH = "./vqa_baseline_bert_finetune/vqa_baseline_bert_finetune.pt"
ANSWER2ID_PATH = "./vqa_baseline_bert_finetune/answer2id.json"

GRADCAM_OUT_DIR = "./gradcam_baseline_bert"
os.makedirs(GRADCAM_OUT_DIR, exist_ok=True)

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class VQABaselineBERT(nn.Module):
    def __init__(self, num_answers, bert_name=BERT_NAME):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, 512)

        self.bert = AutoModel.from_pretrained(bert_name)
        hidden_size = self.bert.config.hidden_size
        self.text_proj = nn.Linear(hidden_size, 512)

        self.classifier = nn.Linear(512, num_answers)

    def forward(self, images, input_ids, attention_mask):
        # (B, 512)
        img_feat = self.cnn(images)

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = out.last_hidden_state[:, 0, :]  # (B, hidden)
        text_feat = self.text_proj(cls)       # (B, 512)

        joint = img_feat + text_feat          # (B, 512)

        logits = self.classifier(joint)       # (B, num_answers)
        return logits



# =================== 2. Grad-CAM 核心类 ===================

class GradCAM:
    def __init__(self, model: VQABaselineBERT, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, images, input_ids, attention_mask, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        images = images.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        logits = self.model(images, input_ids, attention_mask)  # (1, C)
        probs = F.softmax(logits, dim=-1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        score = logits[0, target_class]
        score.backward()

        gradients = self.gradients[0]   # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        alpha = gradients.mean(dim=(1, 2))  # (C,)

        weighted = (alpha.view(-1, 1, 1) * activations).sum(dim=0)  # (H, W)

        cam = F.relu(weighted)

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam_np = cam.cpu().numpy()
        return cam_np, probs[0, target_class].item()



def load_answer_mapping():
    with open(ANSWER2ID_PATH, "r", encoding="utf-8") as f:
        answer2id = json.load(f)
    id2answer = {int(v): k for k, v in answer2id.items()}
    return answer2id, id2answer


def build_model_and_tokenizer():
    answer2id, id2answer = load_answer_mapping()
    num_answers = len(answer2id)

    print("num_answers =", num_answers)

    model = VQABaselineBERT(num_answers=num_answers, bert_name=BERT_NAME)
    model.to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)

    return model, tokenizer, id2answer


def preprocess_sample(example, tokenizer):
    image: Image.Image = example["image"]
    question: str = example["question"]
    answer: str = example["answer"]

    img_tensor = IMG_TRANSFORM(image).unsqueeze(0)  # (1, 3, 224, 224)

    encoded = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    return img_tensor, input_ids, attention_mask, question, answer



def save_gradcam_figure(orig_pil: Image.Image, cam: np.ndarray, out_path: str,
                        title: str = ""):
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    cam_upsampled = F.interpolate(
        cam_tensor,
        size=orig_pil.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(orig_pil.convert("RGB"), cmap="gray")
    plt.imshow(cam_upsampled, cmap="jet", alpha=0.4)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Grad-CAM saved:", out_path)


# =================== 5. 主函数：对几条样本做 Grad-CAM ===================

def run_gradcam_on_examples(num_examples=5):
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    test_set = dataset["test"]
    print("Test sample_num:", len(test_set))

    model, tokenizer, id2answer = build_model_and_tokenizer()
    model.eval()

    target_layer = model.cnn.layer4
    gradcam = GradCAM(model, target_layer)

    indices = list(range(len(test_set)))
    indices = indices[:num_examples]

    for idx in indices:
        example = test_set[idx]
        img_tensor, input_ids, attention_mask, question, gt_answer = preprocess_sample(example, tokenizer)

        #  Grad-CAM
        cam, prob = gradcam.generate(
            images=img_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_class=None,
        )

        with torch.no_grad():
            logits = model(img_tensor.to(DEVICE),
                           input_ids.to(DEVICE),
                           attention_mask.to(DEVICE))
            pred_id = logits.argmax(dim=1).item()
        pred_answer = id2answer.get(pred_id, "<UNK>")

        print(f"\n idx = {idx}")
        print("Question:", question)
        print("GT Answer:", gt_answer)
        print("Pred Answer:", pred_answer, f"(prob={prob:.3f})")

        out_file = os.path.join(GRADCAM_OUT_DIR, f"gradcam_baseline_idx{idx}.jpg")
        save_gradcam_figure(
            orig_pil=example["image"],
            cam=cam,
            out_path=out_file,
            title=f"Q: {question}\nGT: {gt_answer} | Pred: {pred_answer}",
        )


if __name__ == "__main__":
    run_gradcam_on_examples(num_examples=5)
