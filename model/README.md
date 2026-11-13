# Model Weights Directory — VeriNews AI

This directory houses the custom scratch-built Bidirectional LSTM (**FakeNewsBiLSTM**) weights checkpoint trained for **50 Epochs** on the ISOT Fake News Dataset.

```
model/
├── best_model.pt    # PyTorch 50-epoch checkpoint with state_dict, config & embedded vocab
├── vocab.json       # Vocabulary dictionary mapping words to token indices
└── README.md        # Documentation
```

## Checkpoint Structure:
The checkpoint file `model/best_model.pt` contains:
- `architecture`: `"FakeNewsBiLSTM"`
- `epochs`: `50`
- `best_epoch`: `50`
- `best_val_acc`: `100.0`
- `model_config`:
  - `vocab_size`: Total unique tokens in fitted vocabulary
  - `embed_dim`: `64`
  - `hidden_dim`: `64` (Bidirectional concatenation = `128`)
  - `num_layers`: `2`
  - `bidirectional`: `True`
  - `dropout`: `0.3`
  - `num_classes`: `1`
  - `max_len`: `100`
- `model_state_dict`: Saved weights tensor dictionary
- `vocab`: Embedded vocabulary dictionary mapping word strings to index integers
- `created_at`: UTC timestamp

## Training & Retraining:
To retrain or run training with your own dataset, run:
```bash
python -u train.py --epochs 50 --batch-size 64 --lr 0.001
```
The script loads `data/True.csv` and `data/Fake.csv`, fits the `SimpleTokenizer`, trains for exactly 50 epochs, and automatically updates `model/best_model.pt` and `model/vocab.json`.
