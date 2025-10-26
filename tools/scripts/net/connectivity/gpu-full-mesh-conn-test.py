#!/usr/bin/env python3

"""
GPU Full Mesh Connection Test
Description: Executes network discovery on multiple servers and creates a map
             of server IP:network interface to interface IP addresses
"""

import subprocess
import sys
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class NetworkDiscoveryRunner:
    def __init__(self, servers_list_path: str, net_discovery_script: str, ssh_user: str = "dn"):
        """
        Initialize the network discovery runner
        
        Args:
            servers_list_path: Path to the servers.list file
            net_discovery_script: Path to the net-discovery.sh script
            ssh_user: SSH username for remote access (default: "dn")
        """
        self.servers_list_path = servers_list_path
        self.net_discovery_script = net_discovery_script
        self.ssh_user = ssh_user
        self.interface_map: Dict[str, str] = {}
        self.print_lock = threading.Lock()  # For thread-safe printing
        
    def read_servers_list(self) -> List[str]:
        """
        Read server IP addresses from servers.list file
        
        Returns:
            List of server IP addresses
        """
        servers = []
        try:
            with open(self.servers_list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and header lines
                    if not line or line.startswith('Vendor') or line.startswith('#'):
                        continue
                    # Extract IP address (supports tab-separated or plain IP format)
                    parts = line.split()
                    if parts:
                        # Assume first non-header part is the IP
                        ip = parts[0]
                        # Basic IP validation
                        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ip):
                            servers.append(ip)
            return servers
        except FileNotFoundError:
            print(f"Error: Server list file not found: {self.servers_list_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading servers list: {e}", file=sys.stderr)
            sys.exit(1)
    
    def execute_remote_discovery(self, server_ip: str) -> str:
        """
        Execute net-discovery.sh on a remote server via SSH
        
        Args:
            server_ip: IP address of the remote server
            
        Returns:
            Output of the net-discovery script
        """
        try:
            # First, copy the script to the remote server
            remote_script_path = "/tmp/net-discovery.sh"
            
            print(f"Copying net-discovery.sh to {self.ssh_user}@{server_ip}...", file=sys.stderr)
            scp_cmd = [
                "scp",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                self.net_discovery_script,
                f"{self.ssh_user}@{server_ip}:{remote_script_path}"
            ]
            
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"Warning: Failed to copy script to {server_ip}: {result.stderr}", file=sys.stderr)
                return ""
            
            # Execute the script on the remote server
            print(f"Executing net-discovery.sh on {self.ssh_user}@{server_ip}...", file=sys.stderr)
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                f"{self.ssh_user}@{server_ip}",
                f"sudo bash {remote_script_path}"
            ]
            
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"Warning: Failed to execute script on {server_ip}: {result.stderr}", file=sys.stderr)
                return ""
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            print(f"Warning: Timeout executing discovery on {server_ip}", file=sys.stderr)
            return ""
        except Exception as e:
            print(f"Warning: Error executing discovery on {server_ip}: {e}", file=sys.stderr)
            return ""
    
    def parse_discovery_output(self, server_ip: str, output: str) -> List[Tuple[str, str]]:
        """
        Parse the output of net-discovery.sh to extract network interface and IP mappings
        
        Args:
            server_ip: IP address of the server
            output: Output from net-discovery.sh
            
        Returns:
            List of tuples (network_interface, ip_address)
        """
        mappings = []
        
        # Look for the table section with network interface information
        # The table has format: GPU PCIe | Network IF | IP Address | Speed | Status | RDMA Device | NUMA Node
        lines = output.split('\n')
        
        in_table = False
        for line in lines:
            # Detect when we're in the mapping table
            if 'GPU PCIe' in line and 'Network IF' in line and 'IP Address' in line:
                in_table = True
                continue
            
            # Skip separator lines
            if in_table and '---' in line:
                continue
            
            # Exit table when we hit an empty line or new section
            if in_table and (not line.strip() or line.startswith('═') or 'Network Configuration Details' in line):
                break
            
            # Parse table rows
            if in_table and '|' in line:
                # Split by pipe and clean up
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    # parts[1] is Network IF, parts[2] is IP Address
                    net_if = parts[1]
                    ip_addr = parts[2]
                    
                    # Filter out invalid entries
                    if net_if and net_if != 'N/A' and ip_addr and ip_addr != 'N/A':
                        # Remove CIDR notation if present (e.g., "192.168.1.1/24" -> "192.168.1.1")
                        ip_addr = ip_addr.split('/')[0]
                        mappings.append((net_if, ip_addr))
        
        return mappings
    
    def load_interface_map_from_file(self, map_file: str) -> Dict[str, str]:
        """
        Load interface map from a previously saved file
        
        Args:
            map_file: Path to the interface map file
            
        Returns:
            Dictionary mapping server_ip:interface to interface IP address
        """
        interface_map = {}
        
        try:
            with open(map_file, 'r') as f:
                in_table = False
                for line in f:
                    line = line.strip()
                    
                    # Detect table start
                    if 'Server IP:Interface' in line:
                        in_table = True
                        continue
                    
                    # Skip separator lines
                    if line.startswith('=') or line.startswith('-'):
                        continue
                    
                    # Exit table on empty line or footer
                    if in_table and (not line or 'Total:' in line):
                        break
                    
                    # Parse table rows
                    if in_table and '|' in line:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 2:
                            key = parts[0]  # e.g., "172.30.160.118:enp105s0"
                            ip_addr = parts[1]  # e.g., "172.65.2.24"
                            
                            if key and ip_addr and ':' in key:
                                interface_map[key] = ip_addr
            
            print(f"Loaded {len(interface_map)} interfaces from {map_file}", file=sys.stderr)
            return interface_map
            
        except FileNotFoundError:
            print(f"Error: Interface map file not found: {map_file}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading interface map: {e}", file=sys.stderr)
            sys.exit(1)
    
    def build_interface_map(self, servers: List[str]) -> Dict[str, str]:
        """
        Build a complete map of all server interfaces
        
        Args:
            servers: List of server IP addresses
            
        Returns:
            Dictionary mapping "server_ip:interface" to interface IP address
        """
        interface_map = {}
        
        for server_ip in servers:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Processing server: {server_ip}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            
            output = self.execute_remote_discovery(server_ip)
            
            if not output:
                print(f"No output received from {server_ip}", file=sys.stderr)
                continue
            
            mappings = self.parse_discovery_output(server_ip, output)
            
            if not mappings:
                print(f"No network interfaces found on {server_ip}", file=sys.stderr)
                continue
            
            print(f"Found {len(mappings)} network interface(s) on {server_ip}", file=sys.stderr)
            
            for net_if, ip_addr in mappings:
                key = f"{server_ip}:{net_if}"
                interface_map[key] = ip_addr
                print(f"  - {net_if}: {ip_addr}", file=sys.stderr)
        
        return interface_map
    
    def execute_ping_test(self, server_ip: str, local_ip: str, remote_ip: str) -> bool:
        """
        Execute a ping test from a specific interface on a remote server to a target IP
        
        Args:
            server_ip: IP address of the server to run ping from
            local_ip: Source interface IP address to use for ping
            remote_ip: Target IP to ping
            
        Returns:
            True if ping succeeds, False otherwise
        """
        try:
            # Execute ping via SSH with interface IP binding
            # Using -I with IP address is more reliable than interface name
            # Use -c 1 for 1 ping, -W 1 for 1 second timeout
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=5",
                f"{self.ssh_user}@{server_ip}",
                f"ping -I {local_ip} -c 1 -W 1 {remote_ip} > /dev/null 2>&1"
            ]
            
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            print(f"Warning: Error executing ping from {server_ip}:{local_ip} to {remote_ip}: {e}", file=sys.stderr)
            return False
    
    def test_single_interface(self, server_ip: str, local_interface: str, local_ip: str,
                             all_remote_ips: List[str], ip_to_server: Dict[str, str]) -> Dict[str, str]:
        """
        Test all remote IPs from a single interface (used for interface-level parallelization)
        
        Args:
            server_ip: IP address of the source server
            local_interface: Name of the interface
            local_ip: IP address of the interface
            all_remote_ips: List of all remote IPs to test
            ip_to_server: Mapping of IP address to server IP
            
        Returns:
            Dictionary of ping results for this interface
        """
        # ANSI color codes
        GREEN = '\033[0;32m'
        RED = '\033[0;31m'
        NC = '\033[0m'
        
        interface_results = {}
        
        # Ping all remote IPs (excluding own IP and GPUs on same server)
        for remote_ip in all_remote_ips:
            if remote_ip == local_ip:
                # Skip pinging self
                continue
            
            # Get target server IP
            target_server_ip = ip_to_server.get(remote_ip, "Unknown")
            
            # Skip if target is on the same server (GPUs on same server use internal switch)
            if target_server_ip == server_ip:
                continue
            
            test_key = f"{server_ip}:{local_interface}:{remote_ip}"
            
            # Execute ping test using source IP address
            success = self.execute_ping_test(server_ip, local_ip, remote_ip)
            
            if success:
                interface_results[test_key] = "pass"
                status = f"{GREEN}PASS{NC}"
            else:
                interface_results[test_key] = "fail"
                status = f"{RED}FAIL{NC}"
            
            # Thread-safe printing
            with self.print_lock:
                print(f"{server_ip} : {local_ip} -> {target_server_ip} : {remote_ip} ... {status}", file=sys.stderr)
        
        return interface_results
    
    def test_server_interfaces(self, server_ip: str, interfaces: List[Tuple[str, str]], 
                               all_remote_ips: List[str], ip_to_server: Dict[str, str],
                               parallelize_interfaces: bool = False) -> Dict[str, str]:
        """
        Test all interfaces from a single server (executed in parallel per server)
        
        Args:
            server_ip: IP address of the source server
            interfaces: List of (interface_name, interface_ip) tuples for this server
            all_remote_ips: List of all remote IPs to test
            ip_to_server: Mapping of IP address to server IP
            parallelize_interfaces: If True, test each interface in parallel (useful for single server debug)
            
        Returns:
            Dictionary of ping results for this server
        """
        server_results = {}
        
        if parallelize_interfaces and len(interfaces) > 1:
            # Parallel execution by interface (for single-server testing)
            with ThreadPoolExecutor(max_workers=len(interfaces)) as executor:
                future_to_interface = {
                    executor.submit(
                        self.test_single_interface,
                        server_ip,
                        local_interface,
                        local_ip,
                        all_remote_ips,
                        ip_to_server
                    ): local_interface
                    for local_interface, local_ip in interfaces
                }
                
                for future in as_completed(future_to_interface):
                    interface_name = future_to_interface[future]
                    try:
                        interface_results = future.result()
                        server_results.update(interface_results)
                    except Exception as e:
                        print(f"Error testing interface {interface_name} on {server_ip}: {e}", file=sys.stderr)
        else:
            # Sequential execution by interface (for multi-server testing)
            # ANSI color codes
            GREEN = '\033[0;32m'
            RED = '\033[0;31m'
            NC = '\033[0m'
            
            # For each interface on this server
            for local_interface, local_ip in interfaces:
                # Ping all remote IPs (excluding own IP and GPUs on same server)
                for remote_ip in all_remote_ips:
                    if remote_ip == local_ip:
                        # Skip pinging self
                        continue
                    
                    # Get target server IP
                    target_server_ip = ip_to_server.get(remote_ip, "Unknown")
                    
                    # Skip if target is on the same server (GPUs on same server use internal switch)
                    if target_server_ip == server_ip:
                        continue
                    
                    test_key = f"{server_ip}:{local_interface}:{remote_ip}"
                    
                    # Execute ping test using source IP address
                    success = self.execute_ping_test(server_ip, local_ip, remote_ip)
                    
                    if success:
                        server_results[test_key] = "pass"
                        status = f"{GREEN}PASS{NC}"
                    else:
                        server_results[test_key] = "fail"
                        status = f"{RED}FAIL{NC}"
                    
                    # Thread-safe printing
                    with self.print_lock:
                        print(f"{server_ip} : {local_ip} -> {target_server_ip} : {remote_ip} ... {status}", file=sys.stderr)
        
        return server_results
    
    def run_full_mesh_ping_tests(self, interface_map: Dict[str, str], max_workers: int = None, 
                                  test_server: str = None) -> Dict[str, str]:
        """
        Run full mesh ping tests between all interfaces (parallelized by server)
        
        Args:
            interface_map: Dictionary mapping server_ip:interface to interface IP
            max_workers: Maximum number of parallel workers (default: number of servers)
            test_server: If specified, only test from this server IP (default: test all servers)
            
        Returns:
            Dictionary mapping server:interface:remote_ip to test result (pass/fail)
        """
        print("\n" + "=" * 60, file=sys.stderr)
        print("Starting Full Mesh Ping Tests (Parallel Execution)", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        # Create reverse mapping: interface_ip -> server_ip
        ip_to_server = {}
        for key, ip_addr in interface_map.items():
            server_ip = key.split(':', 1)[0]
            ip_to_server[ip_addr] = server_ip
        
        # Group interfaces by server
        server_interfaces = defaultdict(list)
        for key, ip_addr in interface_map.items():
            server_ip, interface = key.split(':', 1)
            server_interfaces[server_ip].append((interface, ip_addr))
        
        # Filter to specific server if requested
        single_server_mode = False
        if test_server:
            if test_server not in server_interfaces:
                print(f"Error: Server {test_server} not found in interface map", file=sys.stderr)
                print(f"Available servers: {', '.join(sorted(server_interfaces.keys()))}", file=sys.stderr)
                return {}
            print(f"Testing from single server: {test_server}", file=sys.stderr)
            print(f"Using interface-level parallelization ({len(server_interfaces[test_server])} interfaces)", file=sys.stderr)
            server_interfaces = {test_server: server_interfaces[test_server]}
            single_server_mode = True
        
        # Collect all remote IPs (target IPs for pinging)
        all_remote_ips = list(interface_map.values())
        
        # If max_workers not specified, use number of servers
        if max_workers is None:
            max_workers = len(server_interfaces)
        
        if not single_server_mode:
            print(f"Testing {len(server_interfaces)} server(s) in parallel with {max_workers} workers", file=sys.stderr)
        print("", file=sys.stderr)
        
        ping_results = {}
        
        # Use ThreadPoolExecutor to parallelize by server
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit a task for each server
            future_to_server = {
                executor.submit(
                    self.test_server_interfaces, 
                    server_ip, 
                    interfaces, 
                    all_remote_ips, 
                    ip_to_server,
                    single_server_mode  # Enable interface parallelization for single server
                ): server_ip 
                for server_ip, interfaces in server_interfaces.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_server):
                server_ip = future_to_server[future]
                try:
                    server_results = future.result()
                    ping_results.update(server_results)
                except Exception as e:
                    print(f"Error testing server {server_ip}: {e}", file=sys.stderr)
        
        # Calculate statistics
        total_tests = len(ping_results)
        passed_tests = sum(1 for result in ping_results.values() if result == "pass")
        
        print("\n" + "=" * 60, file=sys.stderr)
        print(f"Ping Tests Complete: {passed_tests}/{total_tests} passed", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        return ping_results
    
    def run(self) -> Dict[str, str]:
        """
        Run the complete discovery process
        
        Returns:
            Dictionary mapping server_ip:interface to interface IP addresses
        """
        print("Starting GPU Full Mesh Network Discovery", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        # Read server list
        servers = self.read_servers_list()
        print(f"\nFound {len(servers)} server(s) in {self.servers_list_path}", file=sys.stderr)
        for server in servers:
            print(f"  - {server}", file=sys.stderr)
        
        # Build interface map
        self.interface_map = self.build_interface_map(servers)
        
        return self.interface_map
    
    def print_map(self, interface_map: Dict[str, str]):
        """
        Print the interface map in a readable format
        
        Args:
            interface_map: Dictionary to print
        """
        print("\n" + "=" * 60, file=sys.stderr)
        print("Network Interface Map Complete", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"\nTotal interfaces discovered: {len(interface_map)}\n", file=sys.stderr)
        
        # Print the actual map to stdout
        print("=" * 80)
        print("GPU Network Interface Map")
        print("=" * 80)
        print(f"{'Server IP:Interface':<40} | {'Interface IP Address':<30}")
        print("-" * 80)
        
        # Sort by key for better readability
        for key in sorted(interface_map.keys()):
            print(f"{key:<40} | {interface_map[key]:<30}")
        
        print("=" * 80)
        print(f"\nTotal: {len(interface_map)} interfaces")
        print("=" * 80)
    
    def print_ping_results(self, ping_results: Dict[str, str], interface_map: Dict[str, str]):
        """
        Print the ping test results in a readable format with color coding
        
        Args:
            ping_results: Dictionary mapping test keys to pass/fail results
            interface_map: Dictionary mapping server_ip:interface to interface IP
        """
        # ANSI color codes
        GREEN = '\033[0;32m'
        RED = '\033[0;31m'
        NC = '\033[0m'  # No Color
        
        # Create reverse mapping: interface_ip -> server_ip
        ip_to_server = {}
        for key, ip_addr in interface_map.items():
            server_ip = key.split(':', 1)[0]
            ip_to_server[ip_addr] = server_ip
        
        # Also create full reverse mapping: interface_ip -> (server_ip, local_ip)
        ip_to_full_info = {}
        for key, ip_addr in interface_map.items():
            server_ip = key.split(':', 1)[0]
            ip_to_full_info[ip_addr] = (server_ip, ip_addr)
        
        print("\n" + "=" * 110)
        print("Full Mesh Ping Test Results Summary")
        print("=" * 110)
        print(f"{'Source Server : GPU IP':<35} | {'Target Server : GPU IP':<35} | {'Result':<15}")
        print("-" * 110)
        
        passed = 0
        failed = 0
        
        for test_key in sorted(ping_results.keys()):
            parts = test_key.rsplit(':', 2)
            if len(parts) == 3:
                server_ip, interface, remote_ip = parts
                # Get local IP from interface_map
                local_key = f"{server_ip}:{interface}"
                local_ip = interface_map.get(local_key, "Unknown")
                
                # Get target server IP
                target_server_ip = ip_to_server.get(remote_ip, "Unknown")
                
                source_str = f"{server_ip} : {local_ip}"
                target_str = f"{target_server_ip} : {remote_ip}"
            else:
                source_str = "Unknown"
                target_str = "Unknown"
            
            result = ping_results[test_key]
            
            if result == "pass":
                result_colored = f"{GREEN}PASS{NC}"
                passed += 1
            else:
                result_colored = f"{RED}FAIL{NC}"
                failed += 1
            
            print(f"{source_str:<35} | {target_str:<35} | {result_colored}")
        
        print("=" * 110)
        print(f"\nSummary: {passed} passed, {failed} failed out of {passed + failed} total tests")
        
        if failed == 0:
            print(f"{GREEN}All ping tests passed! ✓{NC}")
        else:
            print(f"{RED}Some ping tests failed! ✗{NC}")
        
        print("=" * 110)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='GPU Full Mesh Connection Test - Discover network interfaces on multiple servers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run discovery only
  %(prog)s servers.list
  
  # Run discovery with custom paths
  %(prog)s -s /path/to/servers.list -d /path/to/net-discovery.sh
  
  # Run discovery and ping tests (parallel by default)
  %(prog)s servers.list --ping-test
  
  # Test from only one specific server (useful for debugging)
  %(prog)s --skip-discovery --ping-test --test-server 172.30.160.118
  
  # Run ping tests with custom parallelism
  %(prog)s servers.list --ping-test --parallel-workers 4
  
  # Skip discovery and load from saved map file, then run ping tests
  %(prog)s --skip-discovery --map-file gpu-interface-map.list --ping-test
  
  # Full command with all options
  %(prog)s --servers servers.list --user dn --ping-test --parallel-workers 7
        """
    )
    
    parser.add_argument(
        'servers_list',
        nargs='?',
        default='servers.list',
        help='Path to servers.list file (default: servers.list)'
    )
    
    parser.add_argument(
        '-s', '--servers',
        dest='servers_list_alt',
        help='Alternative way to specify servers.list path'
    )
    
    parser.add_argument(
        '-d', '--discovery-script',
        dest='discovery_script',
        default=None,
        help='Path to net-discovery.sh script (default: auto-detect)'
    )
    
    parser.add_argument(
        '-u', '--user',
        dest='ssh_user',
        default='dn',
        help='SSH username for remote server access (default: dn)'
    )
    
    parser.add_argument(
        '--ping-test',
        action='store_true',
        help='Run full mesh ping tests between all interfaces'
    )
    
    parser.add_argument(
        '--skip-discovery',
        action='store_true',
        help='Skip network discovery and load from existing map file'
    )
    
    parser.add_argument(
        '--map-file',
        dest='map_file',
        default='gpu-interface-map.list',
        help='Interface map file to load when using --skip-discovery (default: gpu-interface-map.list)'
    )
    
    parser.add_argument(
        '--parallel-workers',
        dest='parallel_workers',
        type=int,
        default=None,
        help='Number of parallel workers for ping tests (default: number of servers)'
    )
    
    parser.add_argument(
        '--test-server',
        dest='test_server',
        default=None,
        help='Test from only this specific server IP (useful for debugging a single server)'
    )
    
    args = parser.parse_args()
    
    # Determine servers list path
    servers_list_path = args.servers_list_alt if args.servers_list_alt else args.servers_list
    
    # Determine net-discovery.sh script path
    if args.discovery_script:
        net_discovery_script = args.discovery_script
    else:
        # Try to find net-discovery.sh relative to this script
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent / 'net-discovery.sh',
            script_dir / 'net-discovery.sh',
            Path.cwd() / 'net-discovery.sh',
            Path('/Users/ahalperin/xai/amd-dev/tools/scripts/net/net-discovery.sh')
        ]
        
        net_discovery_script = None
        for path in possible_paths:
            if path.exists():
                net_discovery_script = str(path)
                break
        
        if not net_discovery_script:
            print("Error: Could not find net-discovery.sh script. Please specify with -d option.", file=sys.stderr)
            sys.exit(1)
    
    # Initialize runner
    runner = NetworkDiscoveryRunner(servers_list_path, net_discovery_script, args.ssh_user)
    
    # Handle skip-discovery mode
    if args.skip_discovery:
        print("Skipping network discovery, loading from file...", file=sys.stderr)
        
        if not os.path.exists(args.map_file):
            print(f"Error: Interface map file not found: {args.map_file}", file=sys.stderr)
            sys.exit(1)
        
        interface_map = runner.load_interface_map_from_file(args.map_file)
        
        # Print loaded map
        runner.print_map(interface_map)
    else:
        # Normal discovery mode - verify files exist
        if not os.path.exists(servers_list_path):
            print(f"Error: Servers list not found: {servers_list_path}", file=sys.stderr)
            sys.exit(1)
        
        if not os.path.exists(net_discovery_script):
            print(f"Error: net-discovery.sh not found: {net_discovery_script}", file=sys.stderr)
            sys.exit(1)
        
        # Run discovery
        interface_map = runner.run()
        
        # Print interface map
        runner.print_map(interface_map)
    
    # Run ping tests if requested
    if args.ping_test:
        if len(interface_map) == 0:
            print("\nError: No interfaces discovered. Cannot run ping tests.", file=sys.stderr)
            sys.exit(1)
        
        ping_results = runner.run_full_mesh_ping_tests(interface_map, 
                                                        max_workers=args.parallel_workers,
                                                        test_server=args.test_server)
        runner.print_ping_results(ping_results, interface_map)
        
        # Exit with error code if any tests failed
        failed_count = sum(1 for result in ping_results.values() if result == "fail")
        if failed_count > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()

