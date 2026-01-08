# 🚀 Start Here - Fresh Project Setup

**Team**: Hackathon-nikita  
**Track**: B - BDH-Driven Continuous Narrative Reasoning  
**KDSH 2026**

---

## ✨ This is a FRESH, Ready-to-Train Project

No pre-trained models, no cached data, no artifacts.  
Anyone can extract and start training immediately!

---

## 🏁 Quick Start (3 Steps)

### Step 1: Setup (2 minutes)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train (2-2.5 hours)

```bash
python main.py train --config config.yaml
```

### Step 3: Predict (1 minute)

```bash
python main.py infer \
  --model models/best_model.pt \
  --test-data data/test/ \
  --output results/predictions.csv
```

**Done!** Check `results/predictions.csv` for your submission.

---

## 📚 Documentation

- **README.md** - Complete project documentation
- **QUICKSTART.md** - Quick setup guide
- **report/REPORT.md** - Technical details
- **SUBMISSION_READY.md** - Submission package info

---

## 📦 What's Included

```
✅ Source code (src/)           - BDH model implementation
✅ Training data (80 examples)  - Pre-processed and ready
✅ Test data (60 examples)      - Ready for inference
✅ Books/ folder                - Source novels
✅ CSV files                    - Original data format
✅ Configuration (config.yaml)  - CPU-optimized settings
✅ Documentation                - Complete guides

❌ No trained models            - Train from scratch
❌ No predictions               - Generate your own
❌ No logs                      - Fresh start
```

---

## 🎯 Key Features

- **BDH Architecture**: Stateful attention with persistent memory
- **Pathway Framework**: Document ingestion and vector storage
- **Long-context**: Handles 100k+ word novels
- **CPU-optimized**: Works on Mac/Linux without GPU
- **Reproducible**: Fixed seeds, documented pipeline

---

## ⚙️ System Requirements

- Python 3.8+
- 8GB RAM (16GB recommended)
- ~6 GB disk space (for training)
- CPU (GPU optional but faster)

---

## 🔧 Configuration

Pre-configured for CPU training. To use GPU:

```yaml
# config.yaml
computation:
  device: "cuda"  # Change from "cpu"
  
training:
  batch_size: 8   # Increase from 2
```

---

## 📊 Expected Results

- **Training time**: 2-2.5 hours (CPU), ~30 min (GPU)
- **Training accuracy**: 70-80%
- **Validation accuracy**: 60-70%
- **Model size**: ~50 MB

---

## 🆘 Troubleshooting

**Problem**: Import errors  
**Solution**: `pip install --upgrade -r requirements.txt`

**Problem**: Out of memory  
**Solution**: In `config.yaml`, set `batch_size: 1`

**Problem**: Slow training  
**Solution**: In `config.yaml`, set `num_layers: 4`, `max_novel_length: 100000`

---

## 📞 Questions?

- Read `README.md` for detailed documentation
- Check `QUICKSTART.md` for setup help
- Review `report/REPORT.md` for technical details

---

## 🎉 Ready to Go!

This project is **submission-ready** and **training-ready**.

Extract → Setup → Train → Submit

**Good luck!** 🏆

---

**Package**: Hackathon-nikita_KDSH_2026_FRESH.zip (78 MB)  
**Status**: ✅ Fresh, Clean, Ready
