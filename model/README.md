# Model Weights Directory

Place your pre-trained DistilBERT sequence classification weights file here:

```
model/
└── best_model.pt
```

## Supported Formats for `best_model.pt`:
1. **PyTorch State Dictionary (`state_dict`)**:
   Standard PyTorch weights saved via:
   ```python
   torch.save(model.state_dict(), "best_model.pt")
   # OR
   torch.save({"state_dict": model.state_dict()}, "best_model.pt")
   ```
   The backend will instantiate `DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)` and load the weights.

2. **Complete PyTorch Model Object**:
   Saved via:
   ```python
   torch.save(model, "best_model.pt")
   ```

## Note on Fallback Mode:
If `best_model.pt` is not present, the FastAPI server will gracefully boot in **simulation / heuristic mode** so you can develop, inspect, and test the full API and frontend without crashes. Once you copy your `best_model.pt` file into this folder and restart the server, the backend will automatically load your fine-tuned DistilBERT model.

