# Processed MultiNLI file with columns: context, question, answer, hallucination (0/1), split (train/val/test)
DATA_CSV_FILE = "multinli_processed_multiple_questions.csv"
NUM_EPOCHS = 1000
CHECKPOINT_DIRECTORY = 'Checkpoint_HalluFormer'
TRAINING_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 1

MODEL_WEIGHTS_PATH = "HalluFormer_Model_Weight/HalluFormer.pth"
