#!/usr/bin/env python3
"""
Main CLI for RCCL parameter optimization.
Orchestrates the optimization loop using Bayesian optimization or other methods.
"""

import argparse
import yaml
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from colorama import init, Fore, Style
from tabulate import tabulate

# Initialize colorama for cross-platform colored output
init()

from optimizer import ParameterOptimizer, EarlyStoppingChecker
from executor import RCCLTestExecutor
from parser import parse_rccl_test_output
from results_db import ResultsDatabase


class OptimizationOrchestrator:
    """Orchestrates the RCCL parameter optimization process."""
    
    def __init__(self, config_path: str, resume_db: Optional[str] = None):
        """Initialize the orchestrator.
        
        Args:
            config_path: Path to configuration YAML file
            resume_db: Optional path to database to resume from
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.test_config = self.config['test_config']
        self.fixed_env_vars = self.config['fixed_env_vars']
        self.optimize_params = self.config['optimize_params']
        self.opt_config = self.config['optimization']
        self.output_config = self.config['output']
        
        # Set up output directory
        output_dir = self.output_config['output_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up database
        db_path = self.run_dir / self.output_config['database']
        self.db = ResultsDatabase(str(db_path))
        
        # Create optimization session
        self.session_id = self.db.create_session(
            config=self.config,
            method=self.opt_config['method'],
            objective=self.opt_config['objective']
        )
        
        # Set up optimizer
        self.optimizer = ParameterOptimizer(
            param_space=self.optimize_params,
            method=self.opt_config['method'],
            n_initial_points=self.opt_config.get('n_initial_points', 10)
        )
        
        # Set up executor
        self.executor = RCCLTestExecutor(
            test_config=self.test_config,
            output_dir=str(self.run_dir / "test_outputs"),
            verbose=self.output_config['verbosity'] > 0
        )
        
        # Set up early stopping
        patience = self.opt_config.get('early_stopping_patience', 15)
        self.early_stopping = EarlyStoppingChecker(patience=patience)
        
        self.verbosity = self.output_config['verbosity']
        self.objective_metric = self.opt_config['objective']
        self.direction = self.opt_config['direction']
        
    def print_header(self):
        """Print optimization header."""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Style.BRIGHT}RCCL Parameter Optimization{Style.RESET_ALL}")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        print(f"{Fore.GREEN}Configuration:{Style.RESET_ALL}")
        print(f"  Test: {self.test_config['test_name']}")
        print(f"  Message size: {self.test_config['minbytes']} - {self.test_config['maxbytes']}")
        print(f"  Processes: {self.test_config['num_processes']}")
        print(f"  Method: {self.opt_config['method']}")
        print(f"  Objective: {self.objective_metric} ({self.direction})")
        print(f"  Max iterations: {self.opt_config['max_iterations']}")
        print(f"  Output directory: {self.run_dir}")
        print()
        
        print(f"{Fore.YELLOW}Optimizing {len(self.optimize_params)} parameters:{Style.RESET_ALL}")
        for param_name, param_config in self.optimize_params.items():
            if param_config['type'] == 'categorical':
                values = ', '.join(str(v) for v in param_config['values'])
                print(f"  {param_name}: [{values}]")
            else:
                print(f"  {param_name}: {param_config['range']}")
        print()
    
    def run_single_iteration(self, iteration: int) -> Optional[float]:
        """Run a single optimization iteration.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Objective value or None if failed
        """
        # Get next parameter configuration
        params = self.optimizer.suggest()
        
        if params is None:
            print(f"{Fore.YELLOW}Optimization complete (grid exhausted){Style.RESET_ALL}")
            return None
        
        # Print iteration header
        print(f"\n{Fore.CYAN}{'─'*80}")
        print(f"{Style.BRIGHT}Iteration {iteration}/{self.opt_config['max_iterations']}{Style.RESET_ALL}")
        print(f"{'─'*80}{Style.RESET_ALL}")
        
        if self.verbosity > 0:
            print(f"\n{Fore.YELLOW}Testing parameters:{Style.RESET_ALL}")
            for param, value in params.items():
                print(f"  {param} = {value}")
            print()
        
        # Execute test
        result = self.executor.execute_test(
            parameters=params,
            fixed_env_vars=self.fixed_env_vars,
            run_id=iteration
        )
        
        # Parse results
        parsed = parse_rccl_test_output(result['stdout'], result['stderr'])
        
        # Prepare database entry
        db_entry = {
            'timestamp': result['timestamp'],
            'test_name': self.test_config['test_name'],
            'parameters': params,
            'env_vars': result['env_vars'],
            'command': result['command'],
            'status': parsed['status'],
            'return_code': result['return_code'],
            'error_message': parsed.get('error_message'),
            'execution_time': result['execution_time'],
            'stdout': result['stdout'],
            'stderr': result['stderr'],
            'num_processes': self.test_config['num_processes']
        }
        
        # Add metrics if successful
        objective_value = None
        if parsed['status'] == 'success' and parsed['metrics']:
            metrics = parsed['metrics']
            db_entry.update({
                'busbw_oop': metrics['busbw_oop'],
                'busbw_ip': metrics['busbw_ip'],
                'algbw_oop': metrics['algbw_oop'],
                'algbw_ip': metrics['algbw_ip'],
                'time_oop': metrics['time_oop'],
                'time_ip': metrics['time_ip'],
                'message_size': metrics['message_size']
            })
            
            # Get objective value
            objective_value = metrics.get(self.objective_metric)
            
            if objective_value is not None:
                # Print results
                print(f"\n{Fore.GREEN}✓ Test completed successfully{Style.RESET_ALL}")
                print(f"  Bus BW (out-of-place): {Fore.GREEN}{metrics['busbw_oop']:.2f} GB/s{Style.RESET_ALL}")
                print(f"  Bus BW (in-place):     {Fore.GREEN}{metrics['busbw_ip']:.2f} GB/s{Style.RESET_ALL}")
                print(f"  Objective ({self.objective_metric}): {Fore.CYAN}{objective_value:.2f}{Style.RESET_ALL}")
                
                # Update optimizer
                self.optimizer.update(params, objective_value)
        else:
            print(f"\n{Fore.RED}✗ Test failed: {parsed.get('error_message', 'Unknown error')}{Style.RESET_ALL}")
        
        # Save to database
        run_id = self.db.insert_run(db_entry)
        
        return objective_value
    
    def print_progress_summary(self, iteration: int):
        """Print progress summary.
        
        Args:
            iteration: Current iteration number
        """
        # Get best run so far
        best_run = self.db.get_best_run(self.objective_metric)
        
        if best_run:
            print(f"\n{Fore.CYAN}{'─'*80}")
            print(f"{Style.BRIGHT}Progress Summary{Style.RESET_ALL}")
            print(f"{'─'*80}{Style.RESET_ALL}")
            print(f"  Iterations completed: {iteration}")
            print(f"  Successful runs: {self.db.get_successful_run_count()}")
            print(f"  Best {self.objective_metric}: {Fore.GREEN}{best_run[self.objective_metric]:.2f} GB/s{Style.RESET_ALL}")
            print(f"  {self.early_stopping.get_status()}")
            
            if self.verbosity > 0:
                print(f"\n  {Fore.YELLOW}Best parameters so far:{Style.RESET_ALL}")
                for param, value in best_run['parameters'].items():
                    print(f"    {param} = {value}")
            print()
    
    def run_optimization(self):
        """Run the main optimization loop."""
        self.print_header()
        
        start_time = time.time()
        max_iterations = self.opt_config['max_iterations']
        
        try:
            for iteration in range(1, max_iterations + 1):
                # Run iteration
                objective_value = self.run_single_iteration(iteration)
                
                # Check for early stopping
                if objective_value is not None:
                    should_stop = self.early_stopping.update(objective_value)
                    
                    if should_stop:
                        print(f"\n{Fore.YELLOW}Early stopping triggered - no improvement for {self.early_stopping.patience} iterations{Style.RESET_ALL}")
                        break
                
                # Print progress every 5 iterations
                if iteration % 5 == 0:
                    self.print_progress_summary(iteration)
                
                # Check if grid search is complete
                if self.optimizer.is_complete():
                    print(f"\n{Fore.GREEN}Grid search complete!{Style.RESET_ALL}")
                    break
            
            # Final summary
            self.print_final_summary(time.time() - start_time)
            
            # Run validation if requested
            if self.opt_config.get('validation_runs', 0) > 0:
                self.run_validation()
            
            # Update session
            best_run = self.db.get_best_run(self.objective_metric)
            if best_run:
                self.db.update_session(
                    self.session_id,
                    end_time=datetime.now().isoformat(),
                    best_value=best_run[self.objective_metric],
                    best_run_id=best_run['id'],
                    total_iterations=iteration,
                    status='completed'
                )
            
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Optimization interrupted by user{Style.RESET_ALL}")
            self.db.update_session(
                self.session_id,
                end_time=datetime.now().isoformat(),
                status='interrupted'
            )
            self.print_final_summary(time.time() - start_time)
        
        except Exception as e:
            print(f"\n\n{Fore.RED}Error during optimization: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            self.db.update_session(
                self.session_id,
                end_time=datetime.now().isoformat(),
                status='failed'
            )
    
    def print_final_summary(self, elapsed_time: float):
        """Print final optimization summary.
        
        Args:
            elapsed_time: Total elapsed time in seconds
        """
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Style.BRIGHT}Optimization Complete{Style.RESET_ALL}")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        total_runs = self.db.get_run_count()
        successful_runs = self.db.get_successful_run_count()
        best_run = self.db.get_best_run(self.objective_metric)
        
        print(f"  Total time: {elapsed_time/60:.1f} minutes")
        print(f"  Total runs: {total_runs}")
        print(f"  Successful runs: {successful_runs}")
        print(f"  Success rate: {successful_runs/total_runs*100:.1f}%")
        print()
        
        if best_run:
            print(f"{Fore.GREEN}{Style.BRIGHT}Best Configuration Found:{Style.RESET_ALL}")
            print(f"  {self.objective_metric}: {Fore.GREEN}{best_run[self.objective_metric]:.2f} GB/s{Style.RESET_ALL}")
            print(f"  Bus BW (out-of-place): {best_run['busbw_oop']:.2f} GB/s")
            print(f"  Bus BW (in-place): {best_run['busbw_ip']:.2f} GB/s")
            print(f"  Alg BW (out-of-place): {best_run['algbw_oop']:.2f} GB/s")
            print(f"  Alg BW (in-place): {best_run['algbw_ip']:.2f} GB/s")
            print()
            
            print(f"{Fore.YELLOW}Optimal Parameters:{Style.RESET_ALL}")
            for param, value in best_run['parameters'].items():
                print(f"  {param} = {value}")
            print()
            
            # Save best config
            best_config_file = self.run_dir / "best_config.txt"
            with open(best_config_file, 'w') as f:
                f.write("# Best RCCL Configuration\n")
                f.write(f"# {self.objective_metric}: {best_run[self.objective_metric]:.2f} GB/s\n\n")
                for param, value in best_run['parameters'].items():
                    f.write(f"export {param}={value}\n")
            
            print(f"  {Fore.CYAN}→ Best configuration saved to: {best_config_file}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}No successful runs found{Style.RESET_ALL}")
        
        print(f"\n  {Fore.CYAN}→ Full results database: {self.run_dir / self.output_config['database']}{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}→ Test outputs: {self.executor.get_output_dir()}{Style.RESET_ALL}")
        print()
    
    def run_validation(self):
        """Run validation tests with best configuration."""
        best_run = self.db.get_best_run(self.objective_metric)
        
        if not best_run:
            print(f"{Fore.YELLOW}No best configuration to validate{Style.RESET_ALL}")
            return
        
        n_runs = self.opt_config['validation_runs']
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Style.BRIGHT}Running Validation ({n_runs} runs){Style.RESET_ALL}")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        
        results = []
        params = best_run['parameters']
        
        for i in range(n_runs):
            print(f"Validation run {i+1}/{n_runs}...")
            result = self.executor.execute_test(
                parameters=params,
                fixed_env_vars=self.fixed_env_vars,
                run_id=f"validation_{i+1}"
            )
            
            parsed = parse_rccl_test_output(result['stdout'], result['stderr'])
            if parsed['status'] == 'success' and parsed['metrics']:
                results.append(parsed['metrics'][self.objective_metric])
        
        if results:
            import numpy as np
            mean = np.mean(results)
            std = np.std(results)
            print(f"\n{Fore.GREEN}Validation Results:{Style.RESET_ALL}")
            print(f"  Mean {self.objective_metric}: {mean:.2f} ± {std:.2f} GB/s")
            print(f"  Min: {min(results):.2f} GB/s")
            print(f"  Max: {max(results):.2f} GB/s")
            print(f"  Runs: {len(results)}/{n_runs}")
        else:
            print(f"{Fore.RED}Validation failed - no successful runs{Style.RESET_ALL}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Optimize RCCL parameters for maximum performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new optimization
  %(prog)s --config config.yaml
  
  # Start with custom iteration count
  %(prog)s --config config.yaml --iterations 100
  
  # Resume from previous run
  %(prog)s --resume results.db

For more information, see README.md
"""
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        help='Override max iterations from config'
    )
    parser.add_argument(
        '--resume',
        help='Resume from existing database'
    )
    parser.add_argument(
        '--method',
        choices=['bayesian', 'random', 'grid'],
        help='Override optimization method from config'
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"{Fore.RED}Error: Config file not found: {args.config}{Style.RESET_ALL}")
        sys.exit(1)
    
    # Create orchestrator
    orchestrator = OptimizationOrchestrator(args.config, args.resume)
    
    # Override config if specified
    if args.iterations:
        orchestrator.opt_config['max_iterations'] = args.iterations
    if args.method:
        orchestrator.opt_config['method'] = args.method
    
    # Run optimization
    orchestrator.run_optimization()


if __name__ == '__main__':
    main()


