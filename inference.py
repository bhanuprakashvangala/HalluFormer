import torch
import torch.nn as nn
import argparse
from transformers import BigBirdTokenizer, BigBirdModel
from constants import *
# Define the model class (same as training)
class HalluFormer(nn.Module):
    def __init__(self, pretrained_model_name="google/bigbird-roberta-base"):
        super(HalluFormer, self).__init__()
        self.bigbird = BigBirdModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bigbird.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Binary classification (0: No Hallucination, 1: Hallucination)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bigbird(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.pooler_output  # Extract [CLS] token embedding
        logits = self.classifier(cls_embedding)
        return logits

# Function to load model
def load_model(model_class, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    
    model = model_class()
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

# Function to perform inference
def predict_hallucination(context, question, answer, model, tokenizer, device):
    input_text = f"{context} [SEP] {question} [SEP] {answer}"

    encoded = tokenizer(
        text=context,
        text_pair=f"{question} [SEP] {answer}",
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    return "Hallucination" if prediction == 1 else "Not a Hallucination"

# Main function to handle user input
def main():
    parser = argparse.ArgumentParser(description="Run HalluFormer inference on given text inputs")
    parser.add_argument("--context", type=str, required=True, help="Context text")
    parser.add_argument("--question", type=str, required=True, help="Question text")
    parser.add_argument("--answer", type=str, required=True, help="Answer text")
    parser.add_argument("--model_path", type=str, required=True, default=MODEL_WEIGHTS_PATH,help="Path to the trained model checkpoint")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
    model = load_model(HalluFormer, args.model_path, device)

    # Run inference
    result = predict_hallucination(args.context, args.question, args.answer, model, tokenizer, device)

    # Print result
    print(f"\nPrediction: {result}")

if __name__ == "__main__":
    main()
