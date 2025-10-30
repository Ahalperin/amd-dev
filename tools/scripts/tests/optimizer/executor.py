#!/usr/bin/env python3
"""
Test executor for running RCCL tests with different parameter configurations.
"""

import subprocess
import time
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path


class RCCLTestExecutor:
    """Executes RCCL tests with specified parameters."""
    
    def __init__(self, test_config: Dict, output_dir: str = None, verbose: bool = True):
        """Initialize the executor.
        
        Args:
            test_config: Test configuration dictionary
            output_dir: Directory for output files
            verbose: Whether to print verbose output
        """
        self.test_config = test_config
        self.verbose = verbose
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"rccl_optimization_{timestamp}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_mpirun_command(self, env_vars: Dict[str, any], run_id: int = 0) -> List[str]:
        """Build the mpirun command with environment variables.
        
        Args:
            env_vars: Dictionary of environment variables to set
            run_id: Run identifier for unique output files
            
        Returns:
            List of command components
        """
        config = self.test_config
        
        # Build base mpirun command
        cmd = [
            'mpirun',
            '--np', str(config['num_processes']),
            '--allow-run-as-root',
            '-H', config['mpi_hosts'],
            '--bind-to', config['bind_to'],
        ]
        
        # Add MPI communication interface settings
        if 'oob_tcp_if' in config:
            cmd.extend(['--mca', 'oob_tcp_if_include', config['oob_tcp_if']])
        if 'btl_tcp_if' in config:
            cmd.extend(['--mca', 'btl_tcp_if_include', config['btl_tcp_if']])
        
        # Add all environment variables with -x flag
        for key, value in env_vars.items():
            cmd.extend(['-x', f'{key}={value}'])
        
        # Build test executable path and arguments
        test_executable = f"/home/dn/amd-dev/amd/rccl-tests/build/{config['test_name']}"
        
        test_args = [
            '-b', str(config['minbytes']),
            '-e', str(config['maxbytes']),
            '-f', str(config['stepfactor']),
            '-g', str(config['gpus_per_rank']),
        ]
        
        # Add optional arguments
        if 'iterations' in config:
            test_args.extend(['-n', str(config['iterations'])])
        if 'warmup_iters' in config:
            test_args.extend(['-w', str(config['warmup_iters'])])
        if 'check_iters' in config:
            test_args.extend(['-c', str(config['check_iters'])])
        
        # Add test executable and arguments
        cmd.append(test_executable)
        cmd.extend(test_args)
        
        return cmd
    
    def execute_test(self, parameters: Dict[str, any], fixed_env_vars: Dict[str, any],
                    run_id: int = 0) -> Dict[str, any]:
        """Execute a single RCCL test with given parameters.
        
        Args:
            parameters: Parameters to optimize (will be set as env vars)
            fixed_env_vars: Fixed environment variables
            run_id: Unique run identifier
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        
        # Combine all environment variables
        env_vars = {**fixed_env_vars, **parameters}
        
        # Build command
        cmd = self.build_mpirun_command(env_vars, run_id)
        cmd_str = ' '.join(cmd)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Run #{run_id}")
            print(f"{'='*80}")
            print("Parameters:")
            for key, value in parameters.items():
                print(f"  {key}={value}")
            print()
        
        # Execute command
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy()  # Inherit parent environment
            )
            
            # Wait for completion with timeout
            timeout = self.test_config.get('timeout', 300)
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
            
            execution_time = time.time() - start_time
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id,
                'parameters': parameters,
                'env_vars': env_vars,
                'command': cmd_str,
                'return_code': return_code,
                'execution_time': execution_time,
                'stdout': stdout,
                'stderr': stderr,
                'status': 'success' if return_code == 0 else 'failed'
            }
            
            # Save outputs to files
            self._save_output(run_id, stdout, stderr, parameters)
            
            if self.verbose:
                if return_code == 0:
                    print(f"✓ Test completed successfully in {execution_time:.1f}s")
                else:
                    print(f"✗ Test failed with return code {return_code}")
            
            return result
            
        except subprocess.TimeoutExpired:
            process.kill()
            execution_time = time.time() - start_time
            
            if self.verbose:
                print(f"✗ Test timed out after {execution_time:.1f}s")
            
            return {
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id,
                'parameters': parameters,
                'env_vars': env_vars,
                'command': cmd_str,
                'return_code': -1,
                'execution_time': execution_time,
                'stdout': '',
                'stderr': f'Timeout after {timeout}s',
                'status': 'timeout'
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            if self.verbose:
                print(f"✗ Test failed with exception: {e}")
            
            return {
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id,
                'parameters': parameters,
                'env_vars': env_vars,
                'command': cmd_str,
                'return_code': -1,
                'execution_time': execution_time,
                'stdout': '',
                'stderr': str(e),
                'status': 'error'
            }
    
    def _save_output(self, run_id: int, stdout: str, stderr: str, parameters: Dict):
        """Save test output to files.
        
        Args:
            run_id: Run identifier
            stdout: Standard output
            stderr: Standard error
            parameters: Test parameters
        """
        # Create run-specific directory
        run_dir = self.output_dir / f"run_{run_id:04d}"
        run_dir.mkdir(exist_ok=True)
        
        # Save stdout
        stdout_file = run_dir / "output.log"
        with open(stdout_file, 'w') as f:
            f.write(stdout)
        
        # Save stderr if not empty
        if stderr:
            stderr_file = run_dir / "error.log"
            with open(stderr_file, 'w') as f:
                f.write(stderr)
        
        # Save parameters
        params_file = run_dir / "parameters.txt"
        with open(params_file, 'w') as f:
            for key, value in parameters.items():
                f.write(f"{key}={value}\n")
    
    def validate_parameters(self, parameters: Dict[str, any]) -> Tuple[bool, Optional[str]]:
        """Validate that parameter combination is valid.
        
        Args:
            parameters: Parameter dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Add any parameter validation logic here
        # For example, checking for incompatible combinations
        
        # Check for known incompatible settings
        if parameters.get('NCCL_PXN_DISABLE') == 1 and parameters.get('RCCL_LL128_FORCE_ENABLE') == 1:
            # This is just an example - adjust based on actual RCCL requirements
            pass  # These might actually be compatible
        
        return True, None
    
    def get_output_dir(self) -> Path:
        """Get the output directory path.
        
        Returns:
            Path to output directory
        """
        return self.output_dir


if __name__ == '__main__':
    # Test the executor
    test_config = {
        'test_name': 'all_reduce_perf',
        'minbytes': '256M',
        'maxbytes': '256M',
        'stepfactor': 2,
        'gpus_per_rank': 1,
        'iterations': 20,
        'warmup_iters': 5,
        'check_iters': 1,
        'mpi_hosts': '172.30.160.145:8',
        'num_processes': 8,
        'bind_to': 'numa',
        'timeout': 300
    }
    
    fixed_env = {
        'LD_LIBRARY_PATH': '/home/dn/amd-dev/amd/rccl/build/release:/usr/local/lib:',
        'NCCL_DEBUG': 'VERSION',
    }
    
    parameters = {
        'NCCL_IB_QPS_PER_CONNECTION': 1,
        'NCCL_IB_TC': 104,
        'RCCL_LL128_FORCE_ENABLE': 1,
    }
    
    executor = RCCLTestExecutor(test_config)
    cmd = executor.build_mpirun_command({**fixed_env, **parameters}, run_id=1)
    print("Command:")
    print(' '.join(cmd))


