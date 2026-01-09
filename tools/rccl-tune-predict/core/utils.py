"""Utility functions for data loading and feature engineering."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Integer encoding for collective types
COLLECTIVE_ENCODING = {
    'all_reduce': 0,
    'all_gather': 1,
    'reduce_scatter': 2,
    'alltoall': 3,
    'broadcast': 4,
    'reduce': 5,
    # Also support _perf suffix variants from sweep data
    'all_reduce_perf': 0,
    'all_gather_perf': 1,
    'reduce_scatter_perf': 2,
    'alltoall_perf': 3,
    'broadcast_perf': 4,
    'reduce_perf': 5,
}

# Integer encoding for algorithms
ALGO_ENCODING = {
    'TREE': 0,
    'RING': 1,
    'COLLNETDIRECT': 2,
    'COLLNETCHAIN': 3,
    'NVLS': 4,
    'NVLSTREE': 5,
}

# Integer encoding for protocols
PROTO_ENCODING = {
    'LL': 0,
    'LL128': 1,
    'SIMPLE': 2,
}

# Reverse mappings for decoding
COLLECTIVE_DECODING = {0: 'all_reduce', 1: 'all_gather', 2: 'reduce_scatter', 
                       3: 'alltoall', 4: 'broadcast', 5: 'reduce'}
ALGO_DECODING = {v: k for k, v in ALGO_ENCODING.items()}
PROTO_DECODING = {v: k for k, v in PROTO_ENCODING.items()}


def encode_categorical(algo: str, proto: str) -> Tuple[int, int]:
    """Encode algo and proto strings to integers.
    
    Args:
        algo: Algorithm name (e.g., 'RING', 'TREE')
        proto: Protocol name (e.g., 'LL', 'LL128', 'SIMPLE')
    
    Returns:
        Tuple of (algo_encoded, proto_encoded)
    """
    algo_enc = ALGO_ENCODING.get(algo.upper(), -1)
    proto_enc = PROTO_ENCODING.get(proto.upper(), -1)
    return algo_enc, proto_enc


def decode_categorical(algo_enc: int, proto_enc: int) -> Tuple[str, str]:
    """Decode integer algo and proto back to strings.
    
    Args:
        algo_enc: Encoded algorithm
        proto_enc: Encoded protocol
    
    Returns:
        Tuple of (algo_name, proto_name)
    """
    algo = ALGO_DECODING.get(algo_enc, 'UNKNOWN')
    proto = PROTO_DECODING.get(proto_enc, 'UNKNOWN')
    return algo, proto


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataframe.
    
    Args:
        df: DataFrame with columns: num_nodes, num_gpus, size_bytes
    
    Returns:
        DataFrame with additional columns: log_size, gpus_per_node, bytes_per_gpu
    """
    df = df.copy()
    
    # Log-transform message size for better scaling
    df['log_size'] = np.log2(df['size_bytes'].clip(lower=1))
    
    # GPUs per node
    df['gpus_per_node'] = df['num_gpus'] / df['num_nodes'].clip(lower=1)
    
    # Bytes per GPU
    df['bytes_per_gpu'] = df['size_bytes'] / df['num_gpus'].clip(lower=1)
    
    return df


def load_sweep_data(csv_path: str) -> pd.DataFrame:
    """Load sweep data from CSV and prepare features.
    
    Args:
        csv_path: Path to the metrics CSV file
    
    Returns:
        DataFrame with all features prepared for training
    """
    df = pd.read_csv(csv_path)
    
    # Encode categorical variables
    df['collective_encoded'] = df['collective'].apply(lambda x: COLLECTIVE_ENCODING.get(x.lower(), -1))
    df['algo_encoded'] = df['algo'].apply(lambda x: ALGO_ENCODING.get(x.upper(), -1))
    df['proto_encoded'] = df['proto'].apply(lambda x: PROTO_ENCODING.get(x.upper(), -1))
    
    # Add derived features
    df = engineer_features(df)
    
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from dataframe.
    
    Args:
        df: DataFrame with all prepared features
    
    Returns:
        Tuple of (X, y) where X is feature matrix and y is target (busbw_ip)
    """
    feature_cols = [
        'collective_encoded',
        'num_nodes',
        'num_gpus',
        'size_bytes',
        'log_size',
        'gpus_per_node',
        'bytes_per_gpu',
        'algo_encoded',
        'proto_encoded',
        'nchannels',
    ]
    
    X = df[feature_cols].values
    y = df['busbw_ip'].values
    
    return X, y


def build_feature_vector(
    collective_encoded: int,
    num_nodes: int,
    num_gpus: int,
    size_bytes: int,
    algo_encoded: int,
    proto_encoded: int,
    nchannels: int,
) -> np.ndarray:
    """Build a single feature vector for prediction.
    
    Args:
        collective_encoded: Encoded collective type
        num_nodes: Number of nodes
        num_gpus: Total number of GPUs
        size_bytes: Message size in bytes
        algo_encoded: Encoded algorithm
        proto_encoded: Encoded protocol
        nchannels: Number of channels
    
    Returns:
        1D numpy array of features
    """
    log_size = np.log2(max(size_bytes, 1))
    gpus_per_node = num_gpus / max(num_nodes, 1)
    bytes_per_gpu = size_bytes / max(num_gpus, 1)
    
    return np.array([
        collective_encoded,
        num_nodes,
        num_gpus,
        size_bytes,
        log_size,
        gpus_per_node,
        bytes_per_gpu,
        algo_encoded,
        proto_encoded,
        nchannels,
    ])


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Human readable string (e.g., '16.0 MB')
    """
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


