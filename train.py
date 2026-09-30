import torch
import torch.nn as nn
from transformers import BigBirdTokenizer, BigBirdModel, AdamW
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import os
from dataset import *
from constants import *

# Detect available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Print available GPU information
print(f"Using {num_gpus} GPU(s)")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# **Modified BigBird Model for Binary Classification**
class HalluFormer(nn.Module):
    def __init__(self, pretrained_model_name="google/bigbird-roberta-base"):
        super(HalluFormer, self).__init__()
        self.bigbird = BigBirdModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bigbird.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Binary classification (2 output logits for classes 0 and 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bigbird(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.pooler_output  # [CLS] token embedding
        logits = self.classifier(cls_embedding)
        return logits

# Initialize model and enable **multi-GPU training**
model = HalluFormer()
if num_gpus > 1:
    model = nn.DataParallel(model)  # Wrap model for parallel training
model.to(device)

# Optimizer and Loss Function
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()  # Suitable for binary classification

# Training Function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Validation Function
def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute Precision, Recall, and F1 Score (binary classification)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    return total_loss / len(val_loader), precision, recall, f1

# Main Training Loop
def main():
    # Create DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)

    # Track Best Model
    best_f1 = 0.0
    checkpoint_dir = CHECKPOINT_DIRECTORY
    
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    num_epochs = NUM_EPOCHS
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

        val_loss, precision, recall, f1 = validate_model(model, valid_loader, criterion, device)
        print(
            f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
        )

        # Save Best Model Based on F1 Score
        if f1 > best_f1:
            best_f1 = f1
            best_model_path = os.path.join(checkpoint_dir, f"best_HalluFormer_model_epoch_{epoch+1}_f1_{f1:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1 score: {f1:.4f}")

    print(f"Training complete. Best F1 score: {best_f1:.4f}")

if __name__ == "__main__":
    main()
