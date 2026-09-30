import torch
import torch.nn as nn
from transformers import BigBirdTokenizer, BigBirdModel, AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import transformers
from constants import DATA_CSV_FILE

# Dataset Class
class HallucinationBinaryDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        """
        Initializes the dataset by tokenizing the inputs.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
            tokenizer: Pretrained BigBird tokenizer.
            max_len (int): Maximum sequence length.
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        context = row['context']
        question = row['question']
        answer = row['answer']
        label = row['hallucination']  # Binary classification (0 or 1)

        # Tokenization with BigBird
        encoded = self.tokenizer(
            text=context,
            text_pair=f"{question} [SEP] {answer}",
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Path to CSV file
csv_file = DATA_CSV_FILE  # Replace with the actual path

# Load the dataset
df = pd.read_csv(csv_file)

# Split into train, validation, and test sets based on the 'split' column
train_df = df[df['split'] == 'train']
valid_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']
# Use BigBird tokenizer
transformers.logging.set_verbosity_error()
tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

# Create dataset instances
train_dataset = HallucinationBinaryDataset(train_df, tokenizer, max_len=512)
valid_dataset = HallucinationBinaryDataset(valid_df, tokenizer, max_len=512)
test_dataset = HallucinationBinaryDataset(test_df, tokenizer, max_len=512)

