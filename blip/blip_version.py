import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from tqdm import tqdm


DEVICE = torch.device("cpu")


MODEL_NAME = "Salesforce/blip-vqa-base"
OUTPUT_DIR = "./blip_medvqa_vqarad"

MAX_Q_LEN = 32
MAX_A_LEN = 8
BATCH_SIZE = 2
EPOCHS = 6




def load_vqa_rad():

    dataset = load_dataset("flaviagiammarino/vqa-rad")
    return dataset




def load_blip():

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)
    return processor, model




class VQARADDataset(torch.utils.data.Dataset):

    def __init__(self, hf_split, processor):
        self.dataset = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        question = example["question"]
        answer = example["answer"]


        encoding = self.processor(
            images=image,
            text=question,
            padding="max_length",
            truncation=True,
            max_length=MAX_Q_LEN,
            return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=MAX_A_LEN,
            return_tensors="pt",
        ).input_ids  # [1, MAX_A_LEN]

        encoding["labels"] = labels

        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)

        return encoding



def eval_exact_match(model, processor, raw_eval_dataset, max_samples=200):
    model.eval()
    device = DEVICE
    model.to(device)

    correct = 0
    total = 0

    for i, example in enumerate(raw_eval_dataset):
        if i >= max_samples:
            break

        image = example["image"]
        question = example["question"]
        gt_answer = example["answer"]

        inputs = processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_length=MAX_A_LEN,
                num_beams=3
            )

        pred = processor.decode(out_ids[0], skip_special_tokens=True)

        pred_norm = pred.strip().lower()
        gt_norm = gt_answer.strip().lower()

        if pred_norm == gt_norm:
            correct += 1
        total += 1

        if (i + 1) % 20 == 0:
            print(f"[Eval {i+1}/{max_samples}] 预测: {pred} | 真实: {gt_answer}")

    acc = correct / total if total > 0 else 0.0
    print(f"\nExact match accuracy on first {total} samples: {acc:.4f}")
    return acc


def train_blip_on_vqarad():
    print("device:", DEVICE)

    raw_dataset = load_vqa_rad()
    train_split = raw_dataset["train"]
    test_split = raw_dataset["test"]



    processor, model = load_blip()
    model.to(DEVICE)


    train_dataset = VQARADDataset(train_split, processor)
    test_dataset = VQARADDataset(test_split, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    torch.manual_seed(42)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_steps = 0

        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")
        for step, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            if (step + 1) % 50 == 0:
                avg_loss = total_loss / total_steps
                print(f"[Step {step+1}] avg_loss: {avg_loss:.4f}")

        avg_loss = total_loss / max(total_steps, 1)
        print(f"Epoch {epoch+1} avg_loss: {avg_loss:.4f}")

        print(f"Epoch {epoch+1} accuracy：")
        eval_exact_match(model, processor, test_split, max_samples=100)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


    final_acc = eval_exact_match(model, processor, test_split, max_samples=200)
    print("final_acc:", final_acc)

    example = test_split[0]
    img = example["image"]
    q = example["question"]
    gt = example["answer"]

    img_path = "example_test_image_blip.jpg"
    if isinstance(img, Image.Image):
        img.save(img_path)
    else:
        img = Image.fromarray(img)
        img.save(img_path)


    model.eval()
    with torch.no_grad():
        demo_inputs = processor(images=img, text=q, return_tensors="pt").to(DEVICE)
        out_ids = model.generate(
            **demo_inputs,
            max_length=MAX_A_LEN,
            num_beams=3
        )
    pred = processor.decode(out_ids[0], skip_special_tokens=True)
    print("answer:", pred)


if __name__ == "__main__":
    train_blip_on_vqarad()
