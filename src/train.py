"""
Training Pipeline for BDH Consistency Classifier
Track B: Train BDH model on narrative consistency task
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import json
import yaml

from src.consistency_classifier import ConsistencyClassifier
from src.data_ingestion import PathwayDataIngestion


class NarrativeDataset(Dataset):
    """PyTorch Dataset for narrative consistency"""
    
    def __init__(self, examples: List[Dict], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Combine backstory and novel
        separator = " [SEP] "
        combined = example['backstory'] + separator + example['novel']
        
        # Tokenize
        encoding = self.tokenizer(
            combined,
            max_length=self.max_length * 4,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(example.get('label', 0), dtype=torch.long),
            'story_id': example.get('story_id', '')
        }


class Trainer:
    """Training orchestrator for BDH model"""
    
    def __init__(self, config: Dict, classifier: ConsistencyClassifier):
        self.config = config
        self.classifier = classifier
        self.model = classifier.model
        self.device = classifier.device
        
        # Training config
        self.num_epochs = int(config['training']['num_epochs'])
        self.batch_size = int(config['training']['batch_size'])
        self.learning_rate = float(config['training']['learning_rate'])
        self.warmup_steps = int(config['training']['warmup_steps'])
        self.gradient_accumulation_steps = int(config['training']['gradient_accumulation_steps'])
        self.max_grad_norm = float(config['training']['max_grad_norm'])
        
        # Paths
        self.models_dir = Path(config['paths']['models_dir'])
        self.logs_dir = Path(config['paths']['logs_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.patience = int(config['training']['patience'])
        self.patience_counter = 0
        
        logger.info("Trainer initialized")
    
    def setup_optimizer(self, num_training_steps: int):
        """Setup optimizer and scheduler"""
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': float(self.config['training']['weight_decay'])
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )
        
        # Scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int
    ) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_long_context=True
            )
            
            loss = outputs['loss'] / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            predictions = outputs['predictions']
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (step + 1),
                'acc': correct / total
            })
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
        
        return metrics
    
    def evaluate(self, eval_loader: DataLoader) -> Dict:
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_long_context=True
                )
                
                total_loss += outputs['loss'].item()
                predictions = outputs['predictions']
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = correct / total
        avg_loss = total_loss / len(eval_loader)
        
        # Calculate F1
        tp = sum(p == 1 and l == 1 for p, l in zip(all_predictions, all_labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(all_predictions, all_labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(all_predictions, all_labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.models_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.models_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset
    ):
        """Main training loop"""
        logger.info("Starting training")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config['computation']['num_workers']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config['computation']['num_workers']
        )
        
        # Setup optimizer
        num_training_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        optimizer, scheduler = self.setup_optimizer(num_training_steps)
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler, epoch)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1_score'])
            
            # Check for best model
            is_best = val_metrics['accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['accuracy']
                self.best_loss = val_metrics['loss']
                self.patience_counter = 0
                logger.info(f"New best model! Accuracy: {self.best_accuracy:.4f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config['training']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save training history
        history_path = self.logs_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training completed!")
        logger.info(f"Best accuracy: {self.best_accuracy:.4f}")
        
        return history


def main():
    """Main training script"""
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    logger.info("Initializing training pipeline")
    
    # Data ingestion
    ingestion = PathwayDataIngestion(config)
    
    # Load dataset
    raw_data_path = config['paths']['raw_data']
    dataset = ingestion.load_dataset(raw_data_path)
    
    if not dataset:
        logger.error("No dataset loaded! Please add data to data/raw/")
        return
    
    # Split dataset
    import random
    random.seed(config['data']['random_seed'])
    random.shuffle(dataset)
    
    train_size = int(len(dataset) * config['data']['train_split'])
    val_size = int(len(dataset) * config['data']['val_split'])
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    logger.info(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Initialize classifier
    classifier = ConsistencyClassifier(config)
    
    # Create datasets
    train_dataset = NarrativeDataset(
        train_data,
        classifier.tokenizer,
        config['bdh']['max_seq_length']
    )
    
    val_dataset = NarrativeDataset(
        val_data,
        classifier.tokenizer,
        config['bdh']['max_seq_length']
    )
    
    # Initialize trainer
    trainer = Trainer(config, classifier)
    
    # Train
    history = trainer.train(train_dataset, val_dataset)
    
    logger.info("Training pipeline completed")


if __name__ == "__main__":
    main()
