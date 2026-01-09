"""Configuration search to find optimal RCCL tuning parameters."""

from typing import Dict, Any, List, Optional
import numpy as np

from .busbw_predictor import BusbwPredictor
from .utils import COLLECTIVE_ENCODING, ALGO_ENCODING, PROTO_ENCODING, decode_categorical


def find_optimal_config(
    model: BusbwPredictor,
    collective: str,
    num_nodes: int,
    num_gpus: int,
    size_bytes: int,
    max_nchannels: int = 32,
    coarse_step: int = 4,
    fine_range: int = 3,
    algos: Optional[List[str]] = None,
    protos: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Find the optimal (algo, proto, nchannels) configuration for given workload.
    
    Uses a coarse-to-fine search strategy:
    1. For each (algo, proto) combination, coarse search over nchannels (step=coarse_step)
    2. Fine-tune nchannels around the best coarse value (step=1)
    3. Return the configuration with the highest predicted busbw
    
    Args:
        model: Trained BusbwPredictor model
        collective: Collective type (e.g., 'all_reduce', 'all_gather')
        num_nodes: Number of nodes
        num_gpus: Total number of GPUs
        size_bytes: Message size in bytes
        max_nchannels: Maximum number of channels to search (default: 32)
        coarse_step: Step size for coarse search (default: 4)
        fine_range: Range around best coarse value for fine search (default: 3)
        algos: List of algorithms to search (default: all known)
        protos: List of protocols to search (default: all known)
    
    Returns:
        Dictionary with keys: algo, proto, nchannels, predicted_busbw
    """
    # Encode collective
    collective_enc = COLLECTIVE_ENCODING.get(collective.lower(), -1)
    if collective_enc == -1:
        raise ValueError(f"Unknown collective: {collective}")
    
    # Default to all known algorithms and protocols
    if algos is None:
        algos = list(ALGO_ENCODING.keys())
    if protos is None:
        protos = list(PROTO_ENCODING.keys())
    
    best_config = None
    best_busbw = -np.inf
    
    for algo in algos:
        algo_enc = ALGO_ENCODING.get(algo.upper())
        if algo_enc is None:
            continue
            
        for proto in protos:
            proto_enc = PROTO_ENCODING.get(proto.upper())
            if proto_enc is None:
                continue
            
            # Coarse search over nchannels
            coarse_nchannels = list(range(1, max_nchannels + 1, coarse_step))
            # Always include max_nchannels if not already included
            if max_nchannels not in coarse_nchannels:
                coarse_nchannels.append(max_nchannels)
            
            best_coarse_nc = 1
            best_coarse_busbw = -np.inf
            
            for nc in coarse_nchannels:
                busbw = model.predict_single(
                    collective_enc, num_nodes, num_gpus, size_bytes,
                    algo_enc, proto_enc, nc
                )
                if busbw > best_coarse_busbw:
                    best_coarse_busbw = busbw
                    best_coarse_nc = nc
            
            # Fine search around best coarse value
            fine_start = max(1, best_coarse_nc - fine_range)
            fine_end = min(max_nchannels, best_coarse_nc + fine_range)
            
            best_fine_nc = best_coarse_nc
            best_fine_busbw = best_coarse_busbw
            
            for nc in range(fine_start, fine_end + 1):
                busbw = model.predict_single(
                    collective_enc, num_nodes, num_gpus, size_bytes,
                    algo_enc, proto_enc, nc
                )
                if busbw > best_fine_busbw:
                    best_fine_busbw = busbw
                    best_fine_nc = nc
            
            # Update global best
            if best_fine_busbw > best_busbw:
                best_busbw = best_fine_busbw
                best_config = {
                    'algo': algo,
                    'proto': proto,
                    'nchannels': best_fine_nc,
                    'predicted_busbw': best_fine_busbw,
                }
    
    return best_config


def find_optimal_configs_batch(
    model: BusbwPredictor,
    collective: str,
    num_nodes: int,
    num_gpus: int,
    sizes: List[int],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Find optimal configs for multiple message sizes.
    
    Args:
        model: Trained BusbwPredictor model
        collective: Collective type (e.g., 'all_reduce', 'all_gather')
        num_nodes: Number of nodes
        num_gpus: Total number of GPUs
        sizes: List of message sizes in bytes
        **kwargs: Additional arguments passed to find_optimal_config
    
    Returns:
        List of config dictionaries, one per size
    """
    results = []
    for size in sizes:
        config = find_optimal_config(model, collective, num_nodes, num_gpus, size, **kwargs)
        if config:
            config['size_bytes'] = size
        results.append(config)
    return results


def search_exhaustive(
    model: BusbwPredictor,
    collective: str,
    num_nodes: int,
    num_gpus: int,
    size_bytes: int,
    max_nchannels: int = 32,
    algos: Optional[List[str]] = None,
    protos: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Exhaustively search all configurations and return sorted by busbw.
    
    Useful for analysis and debugging to see how different configs compare.
    
    Args:
        model: Trained BusbwPredictor model
        collective: Collective type (e.g., 'all_reduce', 'all_gather')
        num_nodes: Number of nodes
        num_gpus: Total number of GPUs
        size_bytes: Message size in bytes
        max_nchannels: Maximum number of channels
        algos: List of algorithms to search
        protos: List of protocols to search
    
    Returns:
        List of all configs sorted by predicted_busbw (descending)
    """
    # Encode collective
    collective_enc = COLLECTIVE_ENCODING.get(collective.lower(), -1)
    if collective_enc == -1:
        raise ValueError(f"Unknown collective: {collective}")
    
    if algos is None:
        algos = list(ALGO_ENCODING.keys())
    if protos is None:
        protos = list(PROTO_ENCODING.keys())
    
    all_configs = []
    
    for algo in algos:
        algo_enc = ALGO_ENCODING.get(algo.upper())
        if algo_enc is None:
            continue
            
        for proto in protos:
            proto_enc = PROTO_ENCODING.get(proto.upper())
            if proto_enc is None:
                continue
            
            for nc in range(1, max_nchannels + 1):
                busbw = model.predict_single(
                    collective_enc, num_nodes, num_gpus, size_bytes,
                    algo_enc, proto_enc, nc
                )
                all_configs.append({
                    'algo': algo,
                    'proto': proto,
                    'nchannels': nc,
                    'predicted_busbw': busbw,
                })
    
    # Sort by predicted_busbw descending
    all_configs.sort(key=lambda x: x['predicted_busbw'], reverse=True)
    
    return all_configs


