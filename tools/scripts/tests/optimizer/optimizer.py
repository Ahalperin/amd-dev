#!/usr/bin/env python3
"""
Optimization engine for RCCL parameter tuning.
Supports Bayesian Optimization, Random Search, and Grid Search.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from skopt import Optimizer as BayesianOptimizer
from skopt.space import Integer, Categorical, Real
import itertools
import random


class ParameterOptimizer:
    """Manages parameter optimization using various strategies."""
    
    def __init__(self, param_space: Dict[str, Dict], method: str = "bayesian",
                 n_initial_points: int = 10, random_state: int = 42):
        """Initialize the optimizer.
        
        Args:
            param_space: Parameter space definition from config
            method: Optimization method ("bayesian", "random", "grid")
            n_initial_points: Number of random initial points for Bayesian opt
            random_state: Random seed for reproducibility
        """
        self.param_space = param_space
        self.method = method
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        
        self.param_names = list(param_space.keys())
        self.param_configs = [param_space[name] for name in self.param_names]
        
        # Set up the optimizer based on method
        if method == "bayesian":
            self._setup_bayesian_optimizer()
        elif method == "random":
            self._setup_random_search()
        elif method == "grid":
            self._setup_grid_search()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        self.iteration = 0
    
    def _setup_bayesian_optimizer(self):
        """Set up Bayesian optimization using scikit-optimize."""
        # Convert parameter space to skopt format
        dimensions = []
        
        for param_name, param_config in zip(self.param_names, self.param_configs):
            param_type = param_config['type']
            
            if param_type == 'categorical':
                # Categorical parameter
                dim = Categorical(param_config['values'], name=param_name)
            elif param_type == 'integer':
                # Integer parameter
                low, high = param_config['range']
                dim = Integer(low, high, name=param_name)
            elif param_type == 'real' or param_type == 'continuous':
                # Real-valued parameter
                low, high = param_config['range']
                dim = Real(low, high, name=param_name)
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            dimensions.append(dim)
        
        # Create Bayesian optimizer
        self.optimizer = BayesianOptimizer(
            dimensions=dimensions,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            acq_func='EI',  # Expected Improvement
            acq_optimizer='sampling'
        )
    
    def _setup_random_search(self):
        """Set up random search."""
        np.random.seed(self.random_state)
        random.seed(self.random_state)
    
    def _setup_grid_search(self):
        """Set up grid search."""
        # Generate all combinations
        param_values = []
        
        for param_config in self.param_configs:
            param_type = param_config['type']
            
            if param_type == 'categorical':
                values = param_config['values']
            elif param_type == 'integer':
                low, high = param_config['range']
                # Generate reasonable number of points (max 10 per dimension)
                step = max(1, (high - low) // 10)
                values = list(range(low, high + 1, step))
            elif param_type == 'real' or param_type == 'continuous':
                low, high = param_config['range']
                # Generate 10 evenly spaced points
                values = list(np.linspace(low, high, 10))
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            param_values.append(values)
        
        # Generate all combinations
        self.grid_combinations = list(itertools.product(*param_values))
        self.grid_index = 0
        
        print(f"Grid search: {len(self.grid_combinations)} total combinations")
    
    def suggest(self) -> Dict[str, Any]:
        """Suggest next parameter configuration to try.
        
        Returns:
            Dictionary mapping parameter names to values
        """
        if self.method == "bayesian":
            return self._suggest_bayesian()
        elif self.method == "random":
            return self._suggest_random()
        elif self.method == "grid":
            return self._suggest_grid()
    
    def _suggest_bayesian(self) -> Dict[str, Any]:
        """Suggest parameters using Bayesian optimization."""
        # Ask optimizer for next point
        next_point = self.optimizer.ask()
        
        # Convert to dictionary
        params = {}
        for name, value in zip(self.param_names, next_point):
            params[name] = value
        
        return params
    
    def _suggest_random(self) -> Dict[str, Any]:
        """Suggest random parameters."""
        params = {}
        
        for param_name, param_config in zip(self.param_names, self.param_configs):
            param_type = param_config['type']
            
            if param_type == 'categorical':
                params[param_name] = random.choice(param_config['values'])
            elif param_type == 'integer':
                low, high = param_config['range']
                params[param_name] = random.randint(low, high)
            elif param_type == 'real' or param_type == 'continuous':
                low, high = param_config['range']
                params[param_name] = random.uniform(low, high)
        
        return params
    
    def _suggest_grid(self) -> Optional[Dict[str, Any]]:
        """Suggest next grid point."""
        if self.grid_index >= len(self.grid_combinations):
            return None  # Grid search complete
        
        values = self.grid_combinations[self.grid_index]
        self.grid_index += 1
        
        params = {}
        for name, value in zip(self.param_names, values):
            params[name] = value
        
        return params
    
    def update(self, parameters: Dict[str, Any], objective_value: float):
        """Update optimizer with observed result.
        
        Args:
            parameters: Parameters that were tested
            objective_value: Observed objective value (to maximize)
        """
        self.iteration += 1
        
        if self.method == "bayesian":
            # Convert parameters back to list in correct order
            param_list = [parameters[name] for name in self.param_names]
            
            # Tell optimizer the result (negate for maximization)
            self.optimizer.tell(param_list, -objective_value)
    
    def get_best(self) -> Tuple[Dict[str, Any], float]:
        """Get the best parameters and value seen so far.
        
        Returns:
            Tuple of (best_parameters, best_value)
        """
        if self.method == "bayesian" and hasattr(self.optimizer, 'Xi'):
            if len(self.optimizer.Xi) > 0:
                best_idx = np.argmin(self.optimizer.yi)
                best_params = self.optimizer.Xi[best_idx]
                best_value = -self.optimizer.yi[best_idx]
                
                params_dict = {}
                for name, value in zip(self.param_names, best_params):
                    params_dict[name] = value
                
                return params_dict, best_value
        
        return None, None
    
    def is_complete(self) -> bool:
        """Check if optimization is complete (for grid search).
        
        Returns:
            True if optimization is complete
        """
        if self.method == "grid":
            return self.grid_index >= len(self.grid_combinations)
        
        return False
    
    def get_progress(self) -> str:
        """Get progress string.
        
        Returns:
            Progress description
        """
        if self.method == "grid":
            return f"{self.grid_index}/{len(self.grid_combinations)}"
        else:
            return f"Iteration {self.iteration}"


class EarlyStoppingChecker:
    """Checks for early stopping based on convergence criteria."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.01):
        """Initialize early stopping checker.
        
        Args:
            patience: Number of iterations without improvement before stopping
            min_delta: Minimum change to count as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('-inf')
        self.iterations_without_improvement = 0
        self.history = []
    
    def update(self, value: float) -> bool:
        """Update with new value and check if should stop.
        
        Args:
            value: New objective value
            
        Returns:
            True if should stop early
        """
        self.history.append(value)
        
        if value > self.best_value + self.min_delta:
            # Improvement
            self.best_value = value
            self.iterations_without_improvement = 0
        else:
            # No improvement
            self.iterations_without_improvement += 1
        
        return self.iterations_without_improvement >= self.patience
    
    def get_status(self) -> str:
        """Get status string.
        
        Returns:
            Status description
        """
        return (f"Best: {self.best_value:.2f}, "
                f"No improvement for {self.iterations_without_improvement} iterations")


if __name__ == '__main__':
    # Test the optimizer
    param_space = {
        'NCCL_IB_QPS_PER_CONNECTION': {
            'type': 'categorical',
            'values': [1, 2, 4, 8]
        },
        'NCCL_IB_TC': {
            'type': 'categorical',
            'values': [104, 106, 160, 192]
        },
        'RCCL_LL128_FORCE_ENABLE': {
            'type': 'categorical',
            'values': [0, 1]
        }
    }
    
    optimizer = ParameterOptimizer(param_space, method="bayesian", n_initial_points=5)
    
    print("Testing Bayesian optimizer:")
    for i in range(10):
        params = optimizer.suggest()
        print(f"  Iteration {i+1}: {params}")
        
        # Simulate objective value
        objective = np.random.random() * 100
        optimizer.update(params, objective)
    
    best_params, best_value = optimizer.get_best()
    print(f"\nBest configuration: {best_params}")
    print(f"Best value: {best_value:.2f}")
