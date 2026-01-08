"""
Utility Functions for KDSH Track B
"""

import json
import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
from loguru import logger


def save_json(data: Dict, path: str):
    """Save dictionary to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {path}")


def load_json(path: str) -> Dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}"


def compute_metrics(predictions: List[int], labels: List[int]) -> Dict:
    """
    Compute classification metrics
    
    Args:
        predictions: List of predicted labels
        labels: List of true labels
        
    Returns:
        Dictionary of metrics
    """
    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(labels)
    accuracy = correct / total if total > 0 else 0
    
    # Confusion matrix
    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn
    }


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_device(device_str: str = "cuda") -> torch.device:
    """Get PyTorch device"""
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def validate_dataset_structure(data_dir: Path) -> bool:
    """
    Validate dataset directory structure
    
    Expected structure:
        data_dir/
            1_novel.txt
            1_backstory.txt
            2_novel.txt
            2_backstory.txt
            ...
    """
    if not data_dir.exists():
        logger.error(f"Dataset directory does not exist: {data_dir}")
        return False
    
    # Find novel files
    novel_files = list(data_dir.glob("*_novel.txt"))
    
    if not novel_files:
        logger.error(f"No novel files found in {data_dir}")
        return False
    
    # Check for matching backstory files
    missing_backstories = []
    for novel_file in novel_files:
        story_id = novel_file.stem.replace("_novel", "")
        backstory_file = data_dir / f"{story_id}_backstory.txt"
        
        if not backstory_file.exists():
            missing_backstories.append(story_id)
    
    if missing_backstories:
        logger.warning(f"Missing backstories for: {missing_backstories}")
    
    logger.info(f"Found {len(novel_files)} novel-backstory pairs")
    return True


def create_submission_package(
    team_name: str,
    results_csv: str,
    report_pdf: str,
    output_dir: str = "."
):
    """
    Create submission ZIP package
    
    Package includes:
    - Code (main.py, src/, config.yaml, requirements.txt)
    - Results (results.csv)
    - Report (report.pdf)
    """
    import zipfile
    from datetime import datetime
    
    output_dir = Path(output_dir)
    zip_filename = f"{team_name}_KDSH_2026.zip"
    zip_path = output_dir / zip_filename
    
    files_to_include = [
        'main.py',
        'config.yaml',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'src/',
        results_csv,
        report_pdf
    ]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in files_to_include:
            item_path = Path(item)
            
            if item_path.is_file():
                zipf.write(item_path, item_path.name)
            elif item_path.is_dir():
                for file in item_path.rglob('*'):
                    if file.is_file() and '__pycache__' not in str(file):
                        zipf.write(file, file.relative_to(item_path.parent))
    
    logger.info(f"Created submission package: {zip_path}")
    logger.info(f"Size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
    
    return zip_path


if __name__ == "__main__":
    # Test utilities
    logger.info("Testing utilities...")
    
    # Test metrics
    preds = [1, 0, 1, 1, 0]
    labels = [1, 0, 0, 1, 0]
    metrics = compute_metrics(preds, labels)
    logger.info(f"Metrics: {metrics}")
    
    # Test device
    device = get_device()
    logger.info(f"Device: {device}")
    
    logger.info("Utilities test complete")
