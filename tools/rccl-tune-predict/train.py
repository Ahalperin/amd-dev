#!/usr/bin/env python3
"""Train the BusbwPredictor model from RCCL sweep data."""

import argparse
import sys
from pathlib import Path

# Add parent directory for imports when running as script
sys.path.insert(0, str(Path(__file__).parent))

from core.busbw_predictor import BusbwPredictor
from core.utils import load_sweep_data, prepare_features


def main():
    parser = argparse.ArgumentParser(
        description='Train ML model to predict RCCL bus bandwidth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from rccl-sweep results (when both tools are in the same parent directory)
  python train.py --data ../rccl-sweep/sweep_results/merged_metrics.csv

  # Train with test split and custom output path
  python train.py --data metrics.csv --output models/my_model.pkl --test-split 0.2

  # Train with custom model parameters
  python train.py --data metrics.csv --n-estimators 300 --max-depth 8
"""
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to the metrics CSV file (e.g., merged_metrics.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models/busbw_model.pkl',
        help='Path to save the trained model (default: models/busbw_model.pkl)'
    )
    parser.add_argument(
        '--test-split', '-t',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing (default: 0.2)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200,
        help='Number of boosting stages (default: 200)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum depth of trees (default: 6)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--show-importance',
        action='store_true',
        help='Show feature importance after training'
    )
    
    args = parser.parse_args()
    
    # Load and prepare data
    print(f"Loading data from {args.data}...")
    try:
        df = load_sweep_data(args.data)
    except FileNotFoundError:
        print(f"Error: File not found: {args.data}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} samples")
    
    # Check for required columns
    required_cols = ['busbw_ip', 'algo', 'proto', 'nchannels']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)
    
    # Prepare features
    X, y = prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    
    # Create and train model
    print(f"\nTraining model...")
    model = BusbwPredictor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    
    metrics = model.fit(X, y, test_split=args.test_split)
    
    # Print metrics
    print(f"\nTraining Results:")
    print(f"  Training samples: {metrics['train_samples']}")
    if metrics['test_samples'] > 0:
        print(f"  Testing samples:  {metrics['test_samples']}")
    print(f"  Train R²:         {metrics['train_r2']:.4f}")
    print(f"  Train MAE:        {metrics['train_mae']:.2f} GB/s")
    if 'test_r2' in metrics:
        print(f"  Test R²:          {metrics['test_r2']:.4f}")
        print(f"  Test MAE:         {metrics['test_mae']:.2f} GB/s")
    
    # Show feature importance
    if args.show_importance:
        print(f"\nFeature Importances:")
        importances = model.get_feature_importances()
        for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {name:20s}: {imp:.4f}")
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"\nSaved model to {args.output}")


if __name__ == '__main__':
    main()


