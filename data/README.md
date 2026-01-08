# Dataset Directory

This directory contains the dataset for the KDSH Track B challenge.

## Structure

```
data/
├── raw/                    # Original dataset
│   ├── train/
│   │   ├── 1_novel.txt
│   │   ├── 1_backstory.txt
│   │   ├── 2_novel.txt
│   │   ├── 2_backstory.txt
│   │   └── labels.json
│   │
│   ├── val/               # Validation set (optional)
│   │   └── ...
│   │
│   └── test/              # Test set
│       └── ...
│
└── processed/             # Processed data (auto-generated)
    └── ...
```

## Dataset Format

### Novel Files

- **Filename**: `{story_id}_novel.txt`
- **Format**: Plain text file containing the complete novel (100k+ words)
- **Encoding**: UTF-8

Example: `1_novel.txt`

### Backstory Files

- **Filename**: `{story_id}_backstory.txt`
- **Format**: Plain text file containing hypothetical character backstory
- **Encoding**: UTF-8

Example: `1_backstory.txt`

### Labels File

- **Filename**: `labels.json`
- **Format**: JSON mapping story IDs to labels

```json
{
  "1": 1,
  "2": 0,
  "3": 1,
  ...
}
```

Where:
- `1` = Consistent (backstory is consistent with novel)
- `0` = Inconsistent (backstory contradicts novel)

## Dataset Statistics

*Update after loading your dataset*

| Split | Examples | Consistent | Inconsistent |
|-------|----------|------------|--------------|
| Train | TBD      | TBD        | TBD          |
| Val   | TBD      | TBD        | TBD          |
| Test  | TBD      | TBD        | TBD          |
| Total | TBD      | TBD        | TBD          |

## Loading Dataset

```python
from src.data_ingestion import PathwayDataIngestion
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize ingestion
ingestion = PathwayDataIngestion(config)

# Load dataset
dataset = ingestion.load_dataset('data/raw/train/')

print(f"Loaded {len(dataset)} examples")
```

## Notes

- Place your downloaded dataset in the appropriate subdirectory (`train/`, `val/`, `test/`)
- The system will automatically process and chunk the novels
- Vector embeddings will be generated on-the-fly during training/inference
- Processed data may be cached in `processed/` directory

## Dataset Source

Dataset provided by: Kharagpur Data Science Hackathon 2026  
Track: B - BDH-Driven Continuous Narrative Reasoning

**Dataset Link**: [Provided in competition]
