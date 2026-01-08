"""
Inference Pipeline for BDH Consistency Classifier
Generate predictions for test data
"""

import os
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from loguru import logger
import yaml
import json

from src.consistency_classifier import ConsistencyClassifier
from src.data_ingestion import PathwayDataIngestion


class InferencePipeline:
    """Inference pipeline for narrative consistency classification"""
    
    def __init__(self, config: Dict, model_path: str):
        self.config = config
        self.results_dir = Path(config['paths']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize classifier
        logger.info("Initializing classifier for inference")
        self.classifier = ConsistencyClassifier(config)
        
        # Load trained model
        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.classifier.load_model(model_path)
        
        self.batch_size = config['inference']['batch_size']
        self.generate_rationale = config['inference']['generate_rationale']
        
        logger.info("InferencePipeline initialized")
    
    def predict_single(
        self,
        story_id: str,
        novel: str,
        backstory: str
    ) -> Dict:
        """
        Make prediction for a single example
        
        Returns:
            Prediction dictionary with story_id, prediction, confidence
        """
        result = self.classifier.classify_single(novel, backstory)
        result['story_id'] = story_id
        
        # Generate rationale if requested (optional for Track B)
        if self.generate_rationale:
            evidence = self.classifier.explain_prediction(
                novel,
                backstory,
                result,
                num_passages=self.config['inference']['num_evidence_passages']
            )
            result['evidence'] = evidence
            
            # Create rationale text
            rationale_parts = []
            for i, ev in enumerate(evidence[:3], 1):  # Top 3 passages
                rationale_parts.append(
                    f"Evidence {i}: {ev['text'][:100]}... (score: {ev['score']:.2f})"
                )
            result['rationale'] = " | ".join(rationale_parts)
        else:
            result['rationale'] = ""
        
        return result
    
    def predict_batch(self, examples: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple examples
        
        Args:
            examples: List of dicts with 'story_id', 'novel', 'backstory'
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Running inference on {len(examples)} examples")
        
        results = []
        
        for example in tqdm(examples, desc="Making predictions"):
            try:
                result = self.predict_single(
                    example['story_id'],
                    example['novel'],
                    example['backstory']
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {example['story_id']}: {e}")
                # Add default prediction on error
                results.append({
                    'story_id': example['story_id'],
                    'prediction': 0,
                    'label': 'Inconsistent',
                    'confidence': 0.5,
                    'rationale': f"Error: {str(e)}",
                    'error': True
                })
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str = None):
        """
        Save predictions to CSV in required format
        
        Format:
            Story ID, Prediction, Rationale
            1, 1, Earlier economic shock makes outcome necessary
            2, 0, Proposed backstory contradicts later actions
        """
        if output_path is None:
            output_path = self.results_dir / "results.csv"
        else:
            output_path = Path(output_path)
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            csv_data.append({
                'Story ID': result['story_id'],
                'Prediction': result['prediction'],
                'Rationale': result.get('rationale', '')
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")
        
        # Also save detailed results as JSON
        json_path = output_path.parent / f"{output_path.stem}_detailed.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to {json_path}")
    
    def run(self, test_data_path: str, output_path: str = None):
        """
        Run complete inference pipeline
        
        Args:
            test_data_path: Path to test data directory
            output_path: Path to save results CSV
        """
        logger.info(f"Running inference pipeline on {test_data_path}")
        
        # Load test data
        ingestion = PathwayDataIngestion(self.config)
        test_examples = ingestion.load_dataset(test_data_path)
        
        if not test_examples:
            logger.error(f"No test data found in {test_data_path}")
            return
        
        logger.info(f"Loaded {len(test_examples)} test examples")
        
        # Make predictions
        results = self.predict_batch(test_examples)
        
        # Calculate statistics
        consistent_count = sum(1 for r in results if r['prediction'] == 1)
        inconsistent_count = sum(1 for r in results if r['prediction'] == 0)
        avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
        
        logger.info(f"\nPrediction Statistics:")
        logger.info(f"  Total: {len(results)}")
        logger.info(f"  Consistent: {consistent_count} ({consistent_count/len(results)*100:.1f}%)")
        logger.info(f"  Inconsistent: {inconsistent_count} ({inconsistent_count/len(results)*100:.1f}%)")
        logger.info(f"  Average Confidence: {avg_confidence:.3f}")
        
        # Save results
        self.save_results(results, output_path)
        
        logger.info("Inference pipeline completed")
        
        return results


def generate_submission(config_path: str, model_path: str, test_data_path: str, output_path: str = None):
    """
    Generate submission file for the competition
    
    Args:
        config_path: Path to config.yaml
        model_path: Path to trained model checkpoint
        test_data_path: Path to test data
        output_path: Path to save results.csv
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    pipeline = InferencePipeline(config, model_path)
    
    # Run inference
    results = pipeline.run(test_data_path, output_path)
    
    return results


def evaluate_on_test_set(
    config_path: str,
    model_path: str,
    test_data_path: str,
    ground_truth_path: str = None
):
    """
    Evaluate model on test set with ground truth labels
    
    Args:
        config_path: Path to config.yaml
        model_path: Path to trained model
        test_data_path: Path to test data
        ground_truth_path: Path to ground truth labels JSON (optional)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run inference
    pipeline = InferencePipeline(config, model_path)
    results = pipeline.run(test_data_path)
    
    # Load ground truth if available
    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
        
        # Match predictions with ground truth
        from src.consistency_classifier import evaluate_predictions
        
        predictions = []
        labels = []
        for result in results:
            story_id = result['story_id']
            if story_id in ground_truth:
                predictions.append(result)
                labels.append(ground_truth[story_id])
        
        # Calculate metrics
        metrics = evaluate_predictions(predictions, labels)
        
        logger.info("\nTest Set Evaluation:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        # Save metrics
        metrics_path = Path(config['paths']['results_dir']) / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    else:
        logger.warning("No ground truth provided, skipping evaluation")
        return None


def main():
    """Main inference script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results.csv')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Path to ground truth labels (for evaluation)')
    
    args = parser.parse_args()
    
    if args.ground_truth:
        # Run evaluation
        evaluate_on_test_set(
            args.config,
            args.model,
            args.test_data,
            args.ground_truth
        )
    else:
        # Run inference only
        generate_submission(
            args.config,
            args.model,
            args.test_data,
            args.output
        )


if __name__ == "__main__":
    main()
