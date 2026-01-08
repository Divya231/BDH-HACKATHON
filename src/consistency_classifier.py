"""
Consistency Classifier
Combines BDH model with Pathway data ingestion for narrative consistency reasoning
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from loguru import logger
from transformers import AutoTokenizer
import numpy as np

from src.bdh_model import BDHForConsistencyClassification, load_bdh_model
from src.data_ingestion import PathwayDataIngestion


class ConsistencyClassifier:
    """
    Main classifier for narrative-backstory consistency
    Track B: BDH-driven reasoning over long contexts
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['computation']['device'])
        
        # Initialize BDH model
        logger.info("Initializing BDH model...")
        self.model = load_bdh_model(
            config['bdh'],
            pretrained_path=config['bdh'].get('pretrained_path')
        )
        self.model.to(self.device)
        
        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize data ingestion
        logger.info("Initializing data ingestion...")
        self.data_ingestion = PathwayDataIngestion(config)
        
        # Configuration
        self.max_length = config['bdh']['max_seq_length']
        self.confidence_threshold = config['inference']['confidence_threshold']
        
        logger.info("ConsistencyClassifier initialized")
    
    def tokenize_texts(
        self,
        novel: str,
        backstory: str,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize novel and backstory for BDH processing
        
        Strategy: Concatenate backstory + [SEP] + novel chunks
        This allows BDH to reason about consistency in context
        """
        max_length = max_length or self.max_length
        
        # Create combined input
        # Format: [BACKSTORY] <SEP> [NOVEL]
        separator = " [SEP] "
        combined_text = backstory + separator + novel
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            max_length=max_length * 4,  # Allow longer sequences for BDH
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def prepare_example(self, example: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare a single example for classification
        Handles long-context processing
        """
        story_id = example['story_id']
        novel = example['novel']
        backstory = example['backstory']
        
        logger.info(f"Preparing example {story_id}")
        
        # Tokenize
        inputs = self.tokenize_texts(novel, backstory)
        
        # Add label if available
        if 'label' in example and example['label'] is not None:
            inputs['labels'] = torch.tensor([example['label']], device=self.device)
        
        return inputs
    
    def classify_single(
        self,
        novel: str,
        backstory: str,
        return_confidence: bool = True
    ) -> Dict:
        """
        Classify consistency of a single novel-backstory pair
        
        Args:
            novel: Full novel text
            backstory: Hypothetical backstory text
            return_confidence: Return confidence scores
            
        Returns:
            Dictionary with prediction and optional confidence
        """
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize inputs
            inputs = self.tokenize_texts(novel, backstory)
            
            # Forward pass through BDH
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                use_long_context=True
            )
            
            # Get prediction
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
            prediction = outputs['predictions'].item()
            
            result = {
                'prediction': prediction,
                'label': 'Consistent' if prediction == 1 else 'Inconsistent'
            }
            
            if return_confidence:
                confidence = probabilities[0, prediction].item()
                result['confidence'] = confidence
                result['probabilities'] = {
                    'inconsistent': probabilities[0, 0].item(),
                    'consistent': probabilities[0, 1].item()
                }
        
        logger.info(f"Prediction: {result['label']} (confidence: {result.get('confidence', 0):.3f})")
        return result
    
    def classify_batch(
        self,
        examples: List[Dict],
        batch_size: int = 1
    ) -> List[Dict]:
        """
        Classify multiple examples in batches
        
        Args:
            examples: List of examples with 'novel' and 'backstory'
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for i in range(0, len(examples), batch_size):
                batch = examples[i:i + batch_size]
                
                for example in batch:
                    result = self.classify_single(
                        example['novel'],
                        example['backstory']
                    )
                    result['story_id'] = example.get('story_id', f"example_{i}")
                    results.append(result)
        
        return results
    
    def explain_prediction(
        self,
        novel: str,
        backstory: str,
        prediction: Dict,
        num_passages: int = 5
    ) -> List[Dict]:
        """
        Generate evidence-based explanation for prediction (optional for Track B)
        Uses Pathway vector store to find relevant passages
        
        Args:
            novel: Full novel text
            backstory: Backstory text
            prediction: Prediction dictionary
            num_passages: Number of evidence passages to extract
            
        Returns:
            List of evidence passages with relevance scores
        """
        logger.info("Generating explanation (optional for Track B)")
        
        # Create vector store for novel
        novel_chunks = self.data_ingestion.chunk_text(novel)
        chunk_metadata = [{'chunk_id': i} for i in range(len(novel_chunks))]
        
        vector_store = self.data_ingestion.setup_vector_store(
            novel_chunks,
            chunk_metadata
        )
        
        # Search for relevant passages related to backstory
        evidence_passages = self.data_ingestion.semantic_search(
            backstory,
            vector_store,
            top_k=num_passages
        )
        
        # Add prediction context
        for passage in evidence_passages:
            passage['prediction'] = prediction['label']
            passage['supports'] = passage['score'] > 0.7  # Threshold for support
        
        return evidence_passages
    
    def save_model(self, save_path: str):
        """Save trained model"""
        logger.info(f"Saving model to {save_path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, save_path)
    
    def load_model(self, load_path: str):
        """Load trained model"""
        logger.info(f"Loading model from {load_path}")
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def evaluate_predictions(predictions: List[Dict], ground_truth: List[int]) -> Dict:
    """
    Evaluate predictions against ground truth
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of true labels
        
    Returns:
        Evaluation metrics
    """
    pred_labels = [p['prediction'] for p in predictions]
    
    # Calculate metrics
    correct = sum(p == g for p, g in zip(pred_labels, ground_truth))
    accuracy = correct / len(ground_truth)
    
    # Confusion matrix
    tp = sum(p == 1 and g == 1 for p, g in zip(pred_labels, ground_truth))
    tn = sum(p == 0 and g == 0 for p, g in zip(pred_labels, ground_truth))
    fp = sum(p == 1 and g == 0 for p, g in zip(pred_labels, ground_truth))
    fn = sum(p == 0 and g == 1 for p, g in zip(pred_labels, ground_truth))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn
        },
        'total_examples': len(ground_truth),
        'correct_predictions': correct
    }
    
    logger.info(f"Evaluation metrics: Accuracy={accuracy:.3f}, F1={f1:.3f}")
    return metrics


def main():
    """Test consistency classifier"""
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize classifier
    classifier = ConsistencyClassifier(config)
    
    # Test example
    test_novel = """
    John had always been afraid of water. As a child, he nearly drowned in a lake.
    Throughout his life, he avoided swimming pools and beaches. Even showers made him anxious.
    """
    
    test_backstory = """
    John grew up as a competitive swimmer, winning several championships in his youth.
    He spent every summer at the beach and loved water sports.
    """
    
    # This should be classified as INCONSISTENT
    result = classifier.classify_single(test_novel, test_backstory)
    logger.info(f"Test result: {result}")


if __name__ == "__main__":
    main() 
