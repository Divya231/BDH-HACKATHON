"""
Main Execution Script for KDSH Track B
BDH-Driven Narrative Consistency Reasoning

Usage:
    # Train model
    python main.py train --config config.yaml
    
    # Run inference
    python main.py infer --config config.yaml --model models/best_model.pt --test-data data/test/
    
    # Generate submission
    python main.py submit --config config.yaml --model models/best_model.pt --test-data data/test/
"""

import argparse
import yaml
from pathlib import Path
from loguru import logger
import sys

# Setup logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


def setup_environment(config: dict):
    """Setup environment and directories"""
    paths = config['paths']
    
    # Create directories
    for key, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment setup complete")


def train_model(args):
    """Train BDH model on narrative consistency task"""
    logger.info("="*60)
    logger.info("KDSH Track B - Training Pipeline")
    logger.info("="*60)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_environment(config)
    
    # Import and run training
    from src.train import main as train_main
    train_main()


def run_inference(args):
    """Run inference on test data"""
    logger.info("="*60)
    logger.info("KDSH Track B - Inference Pipeline")
    logger.info("="*60)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_environment(config)
    
    # Import and run inference
    from src.inference import generate_submission
    
    results = generate_submission(
        config_path=args.config,
        model_path=args.model,
        test_data_path=args.test_data,
        output_path=args.output
    )
    
    logger.info(f"Generated {len(results)} predictions")


def generate_submission_file(args):
    """Generate final submission file"""
    logger.info("="*60)
    logger.info("KDSH Track B - Generating Submission")
    logger.info("="*60)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_environment(config)
    
    # Set output path for submission
    if args.output is None:
        team_name = config['project']['team_name']
        output_path = f"{team_name}_KDSH_2026_results.csv"
    else:
        output_path = args.output
    
    # Import and run inference
    from src.inference import generate_submission
    
    results = generate_submission(
        config_path=args.config,
        model_path=args.model,
        test_data_path=args.test_data,
        output_path=output_path
    )
    
    logger.info(f"✅ Submission file generated: {output_path}")
    logger.info(f"   Total predictions: {len(results)}")


def test_setup(args):
    """Test installation and setup"""
    logger.info("="*60)
    logger.info("KDSH Track B - Testing Setup")
    logger.info("="*60)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("✓ Config loaded successfully")
    
    # Test imports
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        
        import pathway as pw
        logger.info(f"✓ Pathway installed")
        
        from transformers import AutoTokenizer
        logger.info(f"✓ Transformers installed")
        
        from src.bdh_model import BDHForConsistencyClassification
        logger.info(f"✓ BDH model module")
        
        from src.data_ingestion import PathwayDataIngestion
        logger.info(f"✓ Data ingestion module")
        
        from src.consistency_classifier import ConsistencyClassifier
        logger.info(f"✓ Consistency classifier module")
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return
    
    # Test model creation
    try:
        from src.bdh_model import load_bdh_model
        
        test_config = {
            'vocab_size': 50257,
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'ff_dim': 1024,
            'max_seq_length': 512,
            'dropout': 0.1,
            'state_dim': 256,
            'memory_size': 128,
            'sparsity_threshold': 0.1,
            'update_mechanism': 'selective'
        }
        
        model = load_bdh_model(test_config)
        logger.info(f"✓ BDH model created successfully")
        
        # Test forward pass
        import torch
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask)
        logger.info(f"✓ Forward pass successful: {outputs['logits'].shape}")
        
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        return
    
    logger.info("\n" + "="*60)
    logger.info("✅ All tests passed! System is ready.")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="KDSH Track B - BDH-Driven Narrative Consistency Reasoning"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the BDH model')
    train_parser.add_argument('--config', type=str, default='config.yaml',
                             help='Path to configuration file')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on test data')
    infer_parser.add_argument('--config', type=str, default='config.yaml',
                             help='Path to configuration file')
    infer_parser.add_argument('--model', type=str, required=True,
                             help='Path to trained model checkpoint')
    infer_parser.add_argument('--test-data', type=str, required=True,
                             help='Path to test data directory')
    infer_parser.add_argument('--output', type=str, default=None,
                             help='Path to save results')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Generate submission file')
    submit_parser.add_argument('--config', type=str, default='config.yaml',
                              help='Path to configuration file')
    submit_parser.add_argument('--model', type=str, required=True,
                              help='Path to trained model checkpoint')
    submit_parser.add_argument('--test-data', type=str, required=True,
                              help='Path to test data directory')
    submit_parser.add_argument('--output', type=str, default=None,
                              help='Output filename for submission')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test setup and installation')
    test_parser.add_argument('--config', type=str, default='config.yaml',
                            help='Path to configuration file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'infer':
        run_inference(args)
    elif args.command == 'submit':
        generate_submission_file(args)
    elif args.command == 'test':
        test_setup(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
