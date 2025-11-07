#!/usr/bin/env python3

"""
Script to build RCCL on multiple servers
Usage: build-cluster.py -s <servers-list> -b <branch-name>
"""

import argparse
import subprocess
import sys
import threading
from typing import List, Tuple


# Color codes for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color


# Thread-safe printing lock
print_lock = threading.Lock()


def print_info(message: str) -> None:
    """Print info message in green"""
    with print_lock:
        print(f"{Colors.GREEN}[INFO]{Colors.NC} {message}")


def print_error(message: str) -> None:
    """Print error message in red"""
    with print_lock:
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print warning message in yellow"""
    with print_lock:
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def build_on_server(server: str, branch: str, npkit: bool) -> bool:
    """
    Build RCCL on a single server
    
    Args:
        server: Server hostname or IP address
        branch: Git branch name to checkout
        npkit: Whether to enable NPKit profiling
        
    Returns:
        True if build succeeded, False otherwise
    """
    print_info("=" * 48)
    print_info(f"Processing server: {server}")
    print_info("=" * 48)
    
    # Build the command to run on the remote server
    build_flag = "--npkit" if npkit else ""
    
    ssh_commands = f"""
set -e
cd /home/dn/amd-dev
echo "Fetching latest changes..."
git fetch -p
echo "Pulling with rebase..."
git pull --rebase
echo "Checking out branch: {branch}"
git checkout {branch}
echo "Starting build..."
cd /home/dn/amd-dev
./dn/build.sh {build_flag}
"""
    
    try:
        # Execute commands on remote server via SSH
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no", 
             server, ssh_commands],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for build
        )
        
        # Print output from remote server
        if result.stdout:
            with print_lock:
                print(result.stdout)
        
        if result.returncode == 0:
            print_info(f"{Colors.GREEN}✓{Colors.NC} Build completed successfully on {server}")
            return True
        else:
            print_error(f"✗ Build failed on {server}")
            if result.stderr:
                with print_lock:
                    print(result.stderr, file=sys.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"✗ Build timed out on {server}")
        return False
    except subprocess.CalledProcessError as e:
        print_error(f"✗ Build failed on {server}: {e}")
        return False
    except Exception as e:
        print_error(f"✗ Error connecting to {server}: {e}")
        return False
    finally:
        with print_lock:
            print()


def check_local_git_status() -> bool:
    """
    Check if there are uncommitted changes in the local repository
    Ignores untracked files, only checks for modified and staged files.
    
    Returns:
        True if repository is clean, False if there are uncommitted changes
    """
    try:
        # Run git status --porcelain to check for any changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd="/home/dn/amd-dev"
        )
        
        # Filter out untracked files (lines starting with '??')
        # Only consider modified and staged files
        tracked_changes = []
        for line in result.stdout.strip().split('\n'):
            if line and not line.startswith('??'):
                tracked_changes.append(line)
        
        # If there are tracked changes, abort
        if tracked_changes:
            print_error("Uncommitted changes detected in local repository!")
            print_error("Please commit or stash your changes before running this script.")
            print()
            print("Uncommitted changes (tracked files only):")
            for change in tracked_changes:
                print(change)
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to check git status: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking git status: {e}")
        return False


def check_unpushed_commits() -> bool:
    """
    Check if there are local commits that haven't been pushed to origin
    
    Returns:
        True if all commits are pushed, False if there are unpushed commits
    """
    try:
        # First, fetch the latest remote information
        print_info("Fetching latest remote information...")
        fetch_result = subprocess.run(
            ["git", "fetch"],
            capture_output=True,
            text=True,
            cwd="/home/dn/amd-dev"
        )
        
        # Get the current branch name
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd="/home/dn/amd-dev"
        )
        current_branch = branch_result.stdout.strip()
        
        # Check for commits that are in HEAD but not in upstream
        result = subprocess.run(
            ["git", "rev-list", "@{u}..HEAD"],
            capture_output=True,
            text=True,
            cwd="/home/dn/amd-dev"
        )
        
        # If output is not empty, there are unpushed commits
        unpushed_commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        if unpushed_commits and unpushed_commits[0]:
            print_error(f"Unpushed commits detected on branch '{current_branch}'!")
            print_error("Please push your commits to origin before running this script.")
            print()
            
            # Show the unpushed commits
            log_result = subprocess.run(
                ["git", "log", "@{u}..HEAD", "--oneline"],
                capture_output=True,
                text=True,
                cwd="/home/dn/amd-dev"
            )
            print("Unpushed commits:")
            print(log_result.stdout)
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        # If there's no upstream branch, warn but don't fail
        if "no upstream" in str(e.stderr).lower() or "@{u}" in str(e.stderr):
            print_warning(f"No upstream branch configured. Skipping unpushed commits check.")
            return True
        print_error(f"Failed to check for unpushed commits: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking for unpushed commits: {e}")
        return False


def build_worker(server: str, branch: str, npkit: bool, 
                 success_list: List[str], failed_list: List[str],
                 list_lock: threading.Lock) -> None:
    """
    Worker function to build on a server (to be run in a thread)
    
    Args:
        server: Server hostname or IP address
        branch: Git branch name to checkout
        npkit: Whether to enable NPKit profiling
        success_list: Thread-safe list to append successful servers
        failed_list: Thread-safe list to append failed servers
        list_lock: Lock for thread-safe list operations
    """
    if build_on_server(server, branch, npkit):
        with list_lock:
            success_list.append(server)
    else:
        with list_lock:
            failed_list.append(server)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Build RCCL on multiple servers via SSH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Description:
  This script performs pre-flight checks on the local repository:
  1. Checks for uncommitted changes (modified/staged files, ignores untracked)
  2. Checks for unpushed commits to origin
  
  If checks pass, it connects to each server in the provided list via SSH and:
  1. Fetches latest git changes (git fetch -p)
  2. Rebases with remote (git pull --rebase)
  3. Checks out the specified branch (git checkout <branch-name>)
  4. Runs dn/build.sh to build RCCL and dependencies
  
  All servers are built in parallel using separate threads for faster execution.

Examples:
  %(prog)s -s "192.168.1.10,192.168.1.11,192.168.1.12" -b feature/new-optimization
  %(prog)s -s "server1,server2" -b main --npkit
        """
    )
    
    parser.add_argument(
        "-s", "--servers",
        required=True,
        help="Comma-separated list of server IPs or hostnames (e.g., '192.168.1.10,192.168.1.11')"
    )
    
    parser.add_argument(
        "-b", "--branch",
        required=True,
        help="Branch name to checkout and build"
    )
    
    parser.add_argument(
        "--npkit",
        action="store_true",
        help="Enable NPKit profiling support in RCCL build"
    )
    
    args = parser.parse_args()
    
    # Check for uncommitted changes in local repository first
    print_info("Checking local repository for uncommitted changes...")
    if not check_local_git_status():
        print_error("Build aborted due to uncommitted changes in local repository.")
        sys.exit(1)
    print_info("Local repository is clean.")
    print()
    
    # Check for unpushed commits
    if not check_unpushed_commits():
        print_error("Build aborted due to unpushed commits in local repository.")
        sys.exit(1)
    print_info("All commits are pushed to origin.")
    print()
    
    # Parse server list
    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    
    if not servers:
        print_error("No valid servers provided")
        sys.exit(1)
    
    if args.npkit:
        print_info("NPKit profiling will be enabled")
    
    print_info("Starting cluster build process (parallel execution)")
    print_info(f"Branch: {args.branch}")
    print_info(f"Servers: {len(servers)} server(s)")
    print()
    
    # Track success/failure with thread-safe lists
    failed_servers: List[str] = []
    success_servers: List[str] = []
    list_lock = threading.Lock()
    
    # Create and start threads for each server
    threads = []
    for server in servers:
        thread = threading.Thread(
            target=build_worker,
            args=(server, args.branch, args.npkit, success_servers, failed_servers, list_lock),
            name=f"BuildThread-{server}"
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Print summary
    print_info("=" * 48)
    print_info("BUILD SUMMARY")
    print_info("=" * 48)
    print_info(f"Total servers: {len(servers)}")
    print_info(f"Successful: {len(success_servers)}")
    print_error(f"Failed: {len(failed_servers)}")
    
    if success_servers:
        print()
        print_info("Successful servers:")
        for server in success_servers:
            print(f"  ✓ {server}")
    
    if failed_servers:
        print()
        print_error("Failed servers:")
        for server in failed_servers:
            print(f"  ✗ {server}")
        sys.exit(1)
    
    print_info("All builds completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()

