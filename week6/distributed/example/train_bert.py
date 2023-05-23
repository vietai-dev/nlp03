import os
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset

import torch
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast


torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# -----------------------------------------------------------------------------
# Load the dataset
dataset = load_dataset("sepidmnorozy/Vietnamese_sentiment")

# Initialize the model, tokenizer, and training settings
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
config = BertConfig.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Preprocess the data using the datasets library
def tokenize_and_encode(batch):
    encoded = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": batch["label"],
    }

encoded_train_dataset = dataset["train"].map(tokenize_and_encode, batched=True, remove_columns=["text"])
encoded_eval_dataset = dataset["test"].map(tokenize_and_encode, batched=True, remove_columns=["text"])
encoded_train_dataset.set_format("torch")
encoded_eval_dataset.set_format("torch")

# Create the DataLoaders
train_dataloader = DataLoader(
    encoded_train_dataset,
    sampler=RandomSampler(encoded_train_dataset),
    batch_size=64,
    collate_fn=lambda x: {
        "input_ids": torch.stack([sample["input_ids"] for sample in x]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in x]),
        "labels": torch.tensor([sample["labels"] for sample in x]),
    },
)

eval_dataloader = DataLoader(
    encoded_eval_dataset,
    sampler=SequentialSampler(encoded_eval_dataset),
    batch_size=64,
    collate_fn=lambda x: {
        "input_ids": torch.stack([sample["input_ids"] for sample in x]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in x]),
        "labels": torch.tensor([sample["labels"] for sample in x]),
    },
)

mixed_precision_dtype = None # torch.bfloat16
ctx = nullcontext() if mixed_precision_dtype == None else torch.amp.autocast(device_type='cuda', dtype=mixed_precision_dtype)
# Initialize GradScaler for mixed precision training
scaler = GradScaler()

for epoch in range(3):
    # Training
    model.train()
    total_train_loss, total_train_correct = 0, 0
    train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Training]", position=0, leave=True)
    for batch in train_progress_bar:
        input_ids, attention_masks, labels = (batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device))

        optimizer.zero_grad()

        # Use autocast to automatically cast tensor types for mixed precision training
        with ctx:
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = criterion(outputs.logits, labels)
        if mixed_precision_dtype:
            # Scale the loss and backpropagate with the help of GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_train_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        total_train_correct += (preds == labels).sum().item()
        
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_correct / len(encoded_train_dataset)
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_accuracy}")

    
    # Evaluation
    model.eval()
    total_eval_loss, total_eval_correct = 0, 0
    eval_progress_bar = tqdm(eval_dataloader, desc=f"Epoch {epoch + 1} [Evaluation]", position=0, leave=True)
    for batch in eval_progress_bar:
        input_ids, attention_masks, labels = (batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device))

        with torch.no_grad():
            with ctx:
                outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
                loss = criterion(outputs.logits, labels)
        total_eval_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        total_eval_correct += (preds == labels).sum().item()

    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    avg_eval_accuracy = total_eval_correct / len(encoded_eval_dataset)
    print(f"Epoch {epoch + 1}, Evaluation Loss: {avg_eval_loss}, Evaluation Accuracy: {avg_eval_accuracy}")