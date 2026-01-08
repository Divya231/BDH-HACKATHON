# Quick Start Guide - KDSH Track B

## 5-Minute Setup

### Step 1: Install Dependencies (2 min)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Verify Setup (30 sec)
```bash
python main.py test --config config.yaml
```

### Step 3: Train Model (2-2.5 hours)
```bash
python main.py train --config config.yaml
```

### Step 4: Generate Predictions (1 min)
```bash
python main.py infer \
  --model models/best_model.pt \
  --test-data data/test/ \
  --output results/predictions.csv
```

## Output

Check `results/predictions.csv` for final predictions!

## Troubleshooting

**Issue**: Import errors  
**Fix**: `pip install -r requirements.txt`

**Issue**: CUDA not available  
**Fix**: Config already set to CPU

**Issue**: Out of memory  
**Fix**: Reduce `batch_size` in config.yaml

---

For detailed documentation, see `README_SUBMISSION.md`
