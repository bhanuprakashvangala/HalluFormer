# HalluFormer

Code for **HalluFormer: A Transformer-Based Framework for Detecting Hallucination in Large Language Models**
Sajid Mahmud, Pawan Neupane, Joel Selvaraj, Bhanu Prakash Vangala, Jianlin Cheng.
AAAI 2025 Spring Symposium on AI for Engineering and Scientific Discoveries.

HalluFormer treats hallucination detection as classification: given a context, a question and an answer, it predicts
whether the answer is supported by the context. It uses BigBird (`google/bigbird-roberta-base`) so long contexts fit,
with a small classification head on top.

| Test set | Precision | Recall | F1 |
|---|---|---|---|
| MultiNLI (test split) | 0.950 | 0.944 | 0.947 |
| ANAH (held out, 683 rows) | 0.756 | 0.703 | 0.729 |

## Files

```
constants.py   data path, batch sizes, epochs, checkpoint paths
dataset.py     loads the CSV and tokenizes (context, question [SEP] answer) pairs
train.py       trains HalluFormer and keeps the checkpoint with the best validation F1
test.py        precision, recall and F1 on the test split
inference.py   scores new examples with a trained checkpoint
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is strongly recommended.

## Data

Training uses [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) rewritten as one CSV with these columns:

| Column | Meaning |
|---|---|
| `context` | the premise |
| `question`, `answer` | the hypothesis written as a question and answer |
| `hallucination` | 1 if the answer contradicts the context, else 0 |
| `split` | `train`, `val` or `test` |

Put the file next to the scripts and set its name in `DATA_CSV_FILE` in `constants.py`.

## Train, test and run

```bash
python train.py                                    # checkpoints go to Checkpoint_HalluFormer/
python test.py                                     # uses MODEL_WEIGHTS_PATH from constants.py
python inference.py --model_path <checkpoint.pth>
```

`NUM_EPOCHS` in `constants.py` is an upper bound; the best checkpoint by validation F1 is the one to keep.
Trained weights are not included in this repository.

## Related

[HalluMat](https://github.com/bhanuprakashvangala/HalluMat), a materials science hallucination benchmark and detector from the same symposium.

## Citation

```bibtex
@inproceedings{mahmud2025halluformer,
  title     = {HalluFormer: A Transformer-Based Framework for Detecting Hallucination in Large Language Models},
  author    = {Mahmud, Sajid and Neupane, Pawan and Selvaraj, Joel and Vangala, Bhanu Prakash and Cheng, Jianlin},
  booktitle = {AAAI 2025 Spring Symposium on AI for Engineering and Scientific Discoveries},
  year      = {2025}
}
```

## License

MIT. This work comes from the [Bioinformatics and Machine Learning Lab](https://calla.rnet.missouri.edu/cheng/) at the University of Missouri.
