import torch
import torch.nn as nn
from transformers import BigBirdTokenizer, BigBirdModel
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
# from multinli_bigbird_dataset import test_dataset  # Import the test dataset
from dataset import test_dataset
from constants import *
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
class HalluFormer(nn.Module):
    def __init__(self, pretrained_model_name="google/bigbird-roberta-base"):
        super(HalluFormer, self).__init__()
        self.bigbird = BigBirdModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bigbird.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Binary classification (2 output logits)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bigbird(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.pooler_output  # [CLS] token embedding
        logits = self.classifier(cls_embedding)
        return logits
def load_model_single_gpu(model_class, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads a model trained with DataParallel on multiple GPUs onto a single GPU or CPU.

    Args:
        model_class: The model class (should be defined before calling this function).
        checkpoint_path (str): Path to the saved checkpoint.
        device (str): Device to load the model onto ("cuda" or "cpu").

    Returns:
        model: The loaded model.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Remove "module." prefix from state_dict keys if the model was trained using DataParallel
    new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}

    # Initialize model
    model = model_class()
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    print(f"Model loaded successfully on {device}")
    return model
# Load trained model
model_path = MODEL_WEIGHTS_PATH  # Update with actual model path
model = load_model_single_gpu(HalluFormer,model_path,device)

model.to(device)
model.eval()
# Load Test DataLoader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize lists for predictions and labels
all_preds = []
all_labels = []
all_probs = []

# Evaluate model
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Probability of class 1
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Compute Metrics
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

# Compute Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Display Metrics
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["No Hallucination", "Hallucination"], yticklabels=["No Hallucination", "Hallucination"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig('roc_curve.png')
