#!/usr/bin/env python3
"""
Parser for RCCL test output to extract performance metrics.
"""

import re
from typing import Dict, Optional, List, Tuple


class RCCLOutputParser:
    """Parses RCCL test output to extract performance metrics."""
    
    @staticmethod
    def parse_output(stdout: str) -> Dict[str, any]:
        """Parse RCCL test output and extract metrics.
        
        Args:
            stdout: Complete stdout from RCCL test
            
        Returns:
            Dictionary with parsed metrics
        """
        result = {
            'success': False,
            'metrics': [],
            'error_message': None
        }
        
        # Check for common error patterns
        if "error" in stdout.lower() or "failed" in stdout.lower():
            # Look for specific error messages
            error_match = re.search(r'(Error|ERROR|FAILED).*', stdout, re.IGNORECASE)
            if error_match:
                result['error_message'] = error_match.group(0)
        
        # Parse the results table
        # Example line:
        #    268435456      67108864     float     sum      -1   1464.9  183.24  343.57      0   1442.5  186.10  348.93      0
        # Format: size count type redop root time algbw busbw #wrong time algbw busbw #wrong
        
        metrics = []
        in_results = False
        
        for line in stdout.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                # Check if we're at the header line
                if 'size' in line and 'count' in line and 'type' in line:
                    in_results = True
                continue
            
            # Check for errors
            if '#wrong' in line and in_results:
                continue
            
            # Stop parsing after summary
            if 'Out of bounds values' in line or 'Avg bus bandwidth' in line:
                in_results = False
                continue
            
            if in_results:
                # Try to parse the data line
                try:
                    parts = line.split()
                    if len(parts) >= 13:  # Full line with both oop and ip results
                        metric = {
                            'size_bytes': int(parts[0]),
                            'count': int(parts[1]),
                            'type': parts[2],
                            'redop': parts[3],
                            'root': int(parts[4]),
                            # Out-of-place metrics
                            'time_oop_us': float(parts[5]),
                            'algbw_oop_gbps': float(parts[6]),
                            'busbw_oop_gbps': float(parts[7]),
                            'errors_oop': int(parts[8]),
                            # In-place metrics
                            'time_ip_us': float(parts[9]),
                            'algbw_ip_gbps': float(parts[10]),
                            'busbw_ip_gbps': float(parts[11]),
                            'errors_ip': int(parts[12]),
                        }
                        metrics.append(metric)
                        result['success'] = True
                except (ValueError, IndexError) as e:
                    # Skip lines that don't match the expected format
                    continue
        
        result['metrics'] = metrics
        
        # Extract average bus bandwidth if present
        avg_match = re.search(r'Avg bus bandwidth\s*:\s*(\d+\.?\d*)', stdout)
        if avg_match:
            result['avg_busbw'] = float(avg_match.group(1))
        
        return result
    
    @staticmethod
    def extract_best_metrics(parsed_output: Dict) -> Optional[Dict[str, float]]:
        """Extract the best/representative metrics from parsed output.
        
        For tests with multiple message sizes, this typically returns
        metrics for the largest message size.
        
        Args:
            parsed_output: Output from parse_output()
            
        Returns:
            Dictionary with best metrics or None if no valid data
        """
        if not parsed_output['success'] or not parsed_output['metrics']:
            return None
        
        # Get the last metric (usually largest message size)
        last_metric = parsed_output['metrics'][-1]
        
        # Check for errors
        if last_metric['errors_oop'] > 0 or last_metric['errors_ip'] > 0:
            return None
        
        return {
            'busbw_oop': last_metric['busbw_oop_gbps'],
            'busbw_ip': last_metric['busbw_ip_gbps'],
            'algbw_oop': last_metric['algbw_oop_gbps'],
            'algbw_ip': last_metric['algbw_ip_gbps'],
            'time_oop': last_metric['time_oop_us'],
            'time_ip': last_metric['time_ip_us'],
            'message_size': last_metric['size_bytes'],
        }
    
    @staticmethod
    def extract_summary_bandwidth(stdout: str) -> Optional[float]:
        """Extract the average bus bandwidth from test summary.
        
        Args:
            stdout: Complete stdout from RCCL test
            
        Returns:
            Average bus bandwidth in GB/s or None
        """
        avg_match = re.search(r'Avg bus bandwidth\s*:\s*(\d+\.?\d*)', stdout)
        if avg_match:
            return float(avg_match.group(1))
        return None
    
    @staticmethod
    def validate_output(stdout: str) -> Tuple[bool, Optional[str]]:
        """Validate that the test output looks correct.
        
        Args:
            stdout: Complete stdout from RCCL test
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for version info (indicates test started)
        if 'RCCL version' not in stdout and 'NCCL version' not in stdout:
            return False, "No RCCL/NCCL version info found - test may not have started"
        
        # Check for device info
        if 'Using devices' not in stdout:
            return False, "No device information found"
        
        # Check for actual results
        if 'out-of-place' not in stdout or 'in-place' not in stdout:
            return False, "No results table found"
        
        # Check for critical errors
        critical_errors = [
            'Segmentation fault',
            'Bus error',
            'CUDA error',
            'HIP error',
            'Assertion failed',
            'Internal error'
        ]
        
        for error in critical_errors:
            if error in stdout:
                return False, f"Critical error detected: {error}"
        
        return True, None


def parse_rccl_test_output(stdout: str, stderr: str = "") -> Dict[str, any]:
    """High-level function to parse RCCL test output.
    
    Args:
        stdout: Standard output from test
        stderr: Standard error from test (optional)
        
    Returns:
        Dictionary with status, metrics, and any errors
    """
    parser = RCCLOutputParser()
    
    # First validate the output
    is_valid, error_msg = parser.validate_output(stdout)
    
    if not is_valid:
        return {
            'status': 'failed',
            'error_message': error_msg,
            'metrics': None
        }
    
    # Parse the output
    parsed = parser.parse_output(stdout)
    
    if not parsed['success']:
        return {
            'status': 'failed',
            'error_message': parsed.get('error_message', 'Failed to parse results'),
            'metrics': None
        }
    
    # Extract best metrics
    metrics = parser.extract_best_metrics(parsed)
    
    if metrics is None:
        return {
            'status': 'failed',
            'error_message': 'No valid metrics found or errors detected in results',
            'metrics': None
        }
    
    return {
        'status': 'success',
        'error_message': None,
        'metrics': metrics,
        'all_metrics': parsed['metrics'],
        'avg_busbw': parsed.get('avg_busbw')
    }


if __name__ == '__main__':
    # Test the parser with sample output
    sample_output = """
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 268435456 maxBytes 268435456 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version develop:33cc4df
# Using devices
#  Rank  0 Group  0 Pid  78515 on amd-mi355x-ses2-1 device  0 [0000:05:00] AMD Instinct MI355X
RCCL version : 2.26.6-heads/drop/2025-08:c3b8de4
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong                               
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)                                      
   268435456      67108864     float     sum      -1   1464.9  183.24  343.57      0   1442.5  186.10  348.93      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 346.251 
#
"""
    
    result = parse_rccl_test_output(sample_output)
    print("Parse result:")
    print(f"  Status: {result['status']}")
    print(f"  Metrics: {result.get('metrics')}")
    print(f"  Avg BW: {result.get('avg_busbw')}")


