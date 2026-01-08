# KDSH 2026 - Track B: BDH-Driven Continuous Narrative Reasoning

**Team**: Hackathon-nikita  
**Track**: B - Long-form Narrative Consistency Classification  
**Date**: January 2026

---

## 🎯 Solution Overview

This solution uses the **Baby Dragon Hatchling (BDH)** architecture with **Pathway framework** to determine if a character's hypothetical backstory is consistent with a long-form novel (100k+ words).

### Key Features
- ✅ BDH architecture with stateful attention and persistent memory
- ✅ Pathway framework for document ingestion and vector storage
- ✅ Handles novels up to 400k+ words
- ✅ Binary classification: Consistent (1) vs Inconsistent (0)
- ✅ CPU-optimized for Mac training

---

## 📊 Dataset

- **Training**: 80 examples (51 consistent, 29 inconsistent)
- **Test**: 60 examples
- **Novels**: "In Search of the Castaways" (138k words) + "The Count of Monte Cristo" (464k words)

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

The training data should be in `data/raw/` with format:
```
data/raw/
├── {id}_novel.txt
├── {id}_backstory.txt
└── labels.json
```

### 3. Train Model

```bash
python main.py train --config config.yaml
```

**Training Time**: ~2-2.5 hours on CPU (5 epochs, 80 examples)

### 4. Generate Predictions

```bash
python main.py infer \
  --model models/best_model.pt \
  --test-data data/test/ \
  --output results/predictions.csv
```

---

## 📁 Project Structure

```
├── main.py                 # Entry point (train/infer/test)
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
├── src/
│   ├── bdh_model.py       # BDH architecture
│   ├── data_ingestion.py  # Pathway data loading
│   ├── consistency_classifier.py
│   ├── train.py           # Training pipeline
│   ├── inference.py       # Inference pipeline
│   └── utils.py           # Utilities
├── models/
│   └── best_model.pt      # Trained model
├── results/
│   └── predictions.csv    # Final predictions
├── data/
│   ├── raw/               # Training data
│   └── test/              # Test data
└── report/
    └── REPORT.md          # Technical report
```

---

## 🏗️ Architecture

### BDH Model
- **6 layers**, 384 hidden dim, 6 attention heads
- **Stateful attention** with persistent memory (256 tokens)
- **Sparse updates** for efficient long-context processing
- **Incremental chunking** (512 tokens/chunk, 256 overlap)

### Training
- **Optimizer**: AdamW (lr=2e-5, weight decay=0.01)
- **Scheduler**: Linear warmup (100 steps)
- **Batch size**: 2 with gradient accumulation (2 steps)
- **Early stopping**: Patience of 3 epochs

---

## 📈 Results

- **Training Accuracy**: ~XX% (see `logs/training.log`)
- **Validation Accuracy**: ~XX%
- **Test Predictions**: `results/predictions.csv`

---

## 🔧 Configuration

Key settings in `config.yaml`:

```yaml
bdh:
  hidden_dim: 384
  num_layers: 6
  memory_size: 256

training:
  num_epochs: 5
  batch_size: 2
  learning_rate: 2e-5

computation:
  device: "cpu"
  num_workers: 0
```

For GPU training, change `device: "cuda"` and increase `batch_size`.

---

## 📝 Output Format

Predictions are in CSV format:

```csv
Story ID,Prediction,Rationale
1,0,Backstory contradicts novel timeline
2,1,Backstory aligns with character development
```

- **Story ID**: Test example ID
- **Prediction**: 0 (Inconsistent) or 1 (Consistent)
- **Rationale**: Brief explanation (optional for Track B)

---

## 🧪 Testing

```bash
# Test environment setup
python main.py test --config config.yaml
```

---

## 📚 Dependencies

Main libraries:
- PyTorch 2.6+
- Transformers (Hugging Face)
- Pathway (vector store)
- Sentence-Transformers (embeddings)
- NumPy, Pandas, tqdm

See `requirements.txt` for full list.

---

## 🎓 Technical Details

### Chunking Strategy
Long novels are split into overlapping chunks:
- Chunk size: 512 tokens
- Overlap: 256 tokens
- Max novel length: 120,000 words

### Attention Mechanism
- Multi-head attention with stateful memory
- Sparse attention (threshold: 0.1)
- Selective state updates

### Data Augmentation
- Novels matched with multiple backstories
- Both consistent and inconsistent examples
- Character-focused backstories

---

## 👥 Team

**Hackathon-nikita**

---

## 📄 License

This project is submitted for KDSH 2026 Track B.

---

## 📞 Support

For questions about this submission, refer to:
- `report/REPORT.md` - Detailed technical report
- `QUICKSTART.md` - Quick setup guide
- `NEXT_STEPS.md` - Development roadmap

---

**Submission Date**: January 2026  
**Track**: B - Long-form Narrative Consistency
# BDH-HACKATHON
