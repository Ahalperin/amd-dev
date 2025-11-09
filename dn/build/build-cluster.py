#!/usr/bin/env python3

"""
Script to build RCCL on multiple servers with comprehensive git verification

This script performs pre-flight checks on local repositories (main, dn/rccl, dn/amd-anp)
to ensure all changes are committed and pushed before starting parallel builds on remote servers.

Branch defaults are read from branch-list.txt and can be overridden via command-line arguments.

Usage:
    build-cluster.py [options]

Optional Arguments:
    -b, --branch BRANCH           Main repository branch (default: from branch-list.txt)
    -s, --servers SERVERS         Comma-separated server list (default: from server-list.txt)
    --rccl-branch BRANCH          RCCL branch/tag (default: from branch-list.txt)
    --amd-anp-branch BRANCH       AMD-ANP branch/tag (default: from branch-list.txt)
    --npkit                       Enable NPKit profiling support in RCCL build

Configuration Files:
    branch-list.txt              Defines default branches for all repositories
                                 Format: repo-name branch-name (one per line)
                                 Example:
                                   amd-dev     main
                                   rccl        drop/2025-08
                                   amd-anp     v1.1.0-5
    
    server-list.txt              List of servers to build on (one IP/hostname per line)

Examples:
    # Use all defaults from branch-list.txt and server-list.txt
    ./build-cluster.py

    # Override only the main branch
    ./build-cluster.py -b feature/test

    # Override RCCL and AMD-ANP branches
    ./build-cluster.py --rccl-branch drop/2025-10 --amd-anp-branch tags/v1.2.0

    # Custom server list with NPKit enabled
    ./build-cluster.py -s "server1,server2" --npkit

    # Full custom configuration
    ./build-cluster.py -s "192.168.1.10,192.168.1.11" -b develop --rccl-branch develop --amd-anp-branch main --npkit

Pre-flight Checks:
    1. Main repository (/home/dn/amd-dev):
       - FAILS if uncommitted changes or unpushed commits touch dn/ directory
       - WARNS (continues) if changes are outside dn/ directory
    2. RCCL repository (dn/rccl): uncommitted changes & unpushed commits (FAILS)
    3. AMD-ANP repository (dn/amd-anp): uncommitted changes & unpushed commits (FAILS)

Remote Build Process:
    1. Fetch latest changes (git fetch -p)
    2. Pull with rebase (git pull --rebase)
    3. Checkout specified branch
    4. Execute build.sh with specified RCCL/AMD-ANP branches
"""

import argparse
import os
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


def build_on_server(server: str, branch: str, npkit: bool, rccl_branch: str, amd_anp_branch: str) -> bool:
    """
    Build RCCL on a single server
    
    Args:
        server: Server hostname or IP address
        branch: Git branch name to checkout
        npkit: Whether to enable NPKit profiling
        rccl_branch: RCCL branch/tag to checkout
        amd_anp_branch: AMD-ANP branch/tag to checkout
        
    Returns:
        True if build succeeded, False otherwise
    """
    print_info("=" * 48)
    print_info(f"Processing server: {server}")
    print_info("=" * 48)
    
    # Build the command to run on the remote server
    npkit_flag = "--npkit" if npkit else ""
    rccl_flag = f"--rccl-branch {rccl_branch}"
    amd_anp_flag = f"--amd-anp-branch {amd_anp_branch}"
    print_info(f"Starting build({npkit_flag}, {rccl_flag}, {amd_anp_flag})")
    ssh_commands = f"""
set -e

cd /home/dn/amd-dev
echo "Fetching latest changes..."
git fetch -p
git checkout {branch}
git pull --rebase

echo "Starting build..."
cd /home/dn/amd-dev
./dn/build/build.sh {npkit_flag} {rccl_flag} {amd_anp_flag}
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
    Only fails if changes are under dn/ directory; warns for other changes.
    
    Returns:
        True if repository is clean or only has changes outside dn/, False if there are uncommitted changes under dn/
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
        # Separate changes under dn/ from other changes
        dn_changes = []
        other_changes = []
        for line in result.stdout.strip().split('\n'):
            if line and not line.startswith('??'):
                # Extract the file path (after the status code)
                # Git status --porcelain format: XY filename
                # where X is index status, Y is working tree status
                # We need to handle formats like " M file", "M  file", "MM file", "R  old -> new"
                parts = line.split(None, 1)  # Split on first whitespace
                if len(parts) >= 2:
                    file_path = parts[1].strip()
                    # Handle renames (format: "old -> new")
                    if ' -> ' in file_path:
                        # For renames, check the new filename
                        file_path = file_path.split(' -> ')[1].strip()
                    
                    if file_path.startswith('dn/'):
                        dn_changes.append(line)
                    else:
                        other_changes.append(line)
        
        # If there are changes under dn/, abort
        if dn_changes:
            print_error("Uncommitted changes detected in local repository under dn/ directory!")
            print_error("Please commit or stash your changes before running this script.")
            print()
            print("Uncommitted changes under dn/ (tracked files only):")
            for change in dn_changes:
                print(change)
            return False
        
        # If there are changes outside dn/, just warn
        if other_changes:
            print_warning("Uncommitted changes detected in local repository (outside dn/ directory):")
            for change in other_changes:
                print(f"  {change}")
            print_warning("These changes are outside dn/ and will not block the build.")
            print()
        
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
    Only fails if unpushed commits touch files under dn/ directory; warns for other commits.
    
    Returns:
        True if all commits are pushed or unpushed commits don't touch dn/, False if there are unpushed commits touching dn/
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
            # Check which files are touched by unpushed commits
            diff_result = subprocess.run(
                ["git", "diff", "--name-only", "@{u}..HEAD"],
                capture_output=True,
                text=True,
                cwd="/home/dn/amd-dev"
            )
            
            changed_files = diff_result.stdout.strip().split('\n') if diff_result.stdout.strip() else []
            
            # Separate files under dn/ from other files
            dn_files = [f for f in changed_files if f.startswith('dn/')]
            other_files = [f for f in changed_files if not f.startswith('dn/')]
            
            # Show the unpushed commits
            log_result = subprocess.run(
                ["git", "log", "@{u}..HEAD", "--oneline"],
                capture_output=True,
                text=True,
                cwd="/home/dn/amd-dev"
            )
            
            if dn_files:
                print_error(f"Unpushed commits detected on branch '{current_branch}' that touch dn/ directory!")
                print_error("Please push your commits to origin before running this script.")
                print()
                print("Unpushed commits:")
                print(log_result.stdout)
                print("Files under dn/ affected by unpushed commits:")
                for f in dn_files:
                    print(f"  {f}")
                return False
            else:
                print_warning(f"Unpushed commits detected on branch '{current_branch}' (not touching dn/ directory):")
                print(log_result.stdout)
                print_warning("These commits don't affect dn/ and will not block the build.")
                print()
        
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


def check_subdir_git_status(repo_path: str, repo_name: str) -> bool:
    """
    Check if there are uncommitted changes in a subdirectory git repository
    
    Args:
        repo_path: Absolute path to the repository
        repo_name: Name of the repository (for display purposes)
        
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
            cwd=repo_path
        )
        
        # Filter out untracked files (lines starting with '??')
        # Only consider modified and staged files
        tracked_changes = []
        for line in result.stdout.strip().split('\n'):
            if line and not line.startswith('??'):
                tracked_changes.append(line)
        
        # If there are tracked changes, abort
        if tracked_changes:
            print_error(f"Uncommitted changes detected in {repo_name} repository!")
            print_error(f"Path: {repo_path}")
            print_error("Please commit or stash your changes before running this script.")
            print()
            print(f"Uncommitted changes in {repo_name} (tracked files only):")
            for change in tracked_changes:
                print(change)
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to check git status for {repo_name}: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking git status for {repo_name}: {e}")
        return False


def check_subdir_unpushed_commits(repo_path: str, repo_name: str) -> bool:
    """
    Check if there are local commits that haven't been pushed to origin in a subdirectory
    
    Args:
        repo_path: Absolute path to the repository
        repo_name: Name of the repository (for display purposes)
        
    Returns:
        True if all commits are pushed, False if there are unpushed commits
    """
    try:
        # First, fetch the latest remote information
        fetch_result = subprocess.run(
            ["git", "fetch"],
            capture_output=True,
            text=True,
            cwd=repo_path
        )
        
        # Get the current branch name
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_path
        )
        current_branch = branch_result.stdout.strip()
        
        # Check for commits that are in HEAD but not in upstream
        result = subprocess.run(
            ["git", "rev-list", "@{u}..HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_path
        )
        
        # If output is not empty, there are unpushed commits
        unpushed_commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        if unpushed_commits and unpushed_commits[0]:
            print_error(f"Unpushed commits detected in {repo_name} on branch '{current_branch}'!")
            print_error(f"Path: {repo_path}")
            print_error("Please push your commits to origin before running this script.")
            print()
            
            # Show the unpushed commits
            log_result = subprocess.run(
                ["git", "log", "@{u}..HEAD", "--oneline"],
                capture_output=True,
                text=True,
                cwd=repo_path
            )
            print(f"Unpushed commits in {repo_name}:")
            print(log_result.stdout)
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        # If there's no upstream branch, warn but don't fail
        if "no upstream" in str(e.stderr).lower() or "@{u}" in str(e.stderr):
            print_warning(f"No upstream branch configured for {repo_name}. Skipping unpushed commits check.")
            return True
        print_error(f"Failed to check for unpushed commits in {repo_name}: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking for unpushed commits in {repo_name}: {e}")
        return False


def build_worker(server: str, branch: str, npkit: bool, rccl_branch: str, amd_anp_branch: str,
                 success_list: List[str], failed_list: List[str],
                 list_lock: threading.Lock) -> None:
    """
    Worker function to build on a server (to be run in a thread)
    
    Args:
        server: Server hostname or IP address
        branch: Git branch name to checkout
        npkit: Whether to enable NPKit profiling
        rccl_branch: RCCL branch/tag to checkout
        amd_anp_branch: AMD-ANP branch/tag to checkout
        success_list: Thread-safe list to append successful servers
        failed_list: Thread-safe list to append failed servers
        list_lock: Lock for thread-safe list operations
    """
    if build_on_server(server, branch, npkit, rccl_branch, amd_anp_branch):
        with list_lock:
            success_list.append(server)
    else:
        with list_lock:
            failed_list.append(server)


def read_servers_from_file(file_path: str) -> List[str]:
    """
    Read server IPs/hostnames from a file (one per line)
    
    Args:
        file_path: Path to the file containing server list
        
    Returns:
        List of server IPs/hostnames
    """
    try:
        with open(file_path, 'r') as f:
            servers = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        return servers
    except FileNotFoundError:
        print_error(f"Server list file not found: {file_path}")
        return []
    except Exception as e:
        print_error(f"Error reading server list file: {e}")
        return []


def read_branches_from_file(file_path: str) -> dict:
    """
    Read branch names from branch-list.txt file
    
    Args:
        file_path: Path to the file containing branch list
        
    Returns:
        Dictionary mapping repo names to branch names
        Example: {'amd-dev': 'main', 'rccl': 'drop/2025-08', 'amd-anp': 'v1.1.0-5'}
    """
    branches = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse the line: repo-name branch-name
                parts = line.split(None, 1)  # Split on whitespace, max 2 parts
                if len(parts) == 2:
                    repo_name, branch_name = parts
                    branches[repo_name] = branch_name
        
        return branches
    except FileNotFoundError:
        print_warning(f"Branch list file not found: {file_path}")
        return {}
    except Exception as e:
        print_warning(f"Error reading branch list file: {e}")
        return {}


def main():
    """Main function"""
    # Read branch defaults from branch-list.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    branch_file = os.path.join(script_dir, "branch-list.txt")
    branch_defaults = read_branches_from_file(branch_file)
    
    # Extract defaults from file or use hardcoded fallbacks
    default_main_branch = branch_defaults.get('amd-dev', 'main')
    default_rccl_branch = branch_defaults.get('rccl', 'drop/2025-08')
    default_amd_anp_branch = branch_defaults.get('amd-anp', 'tags/v1.1.0-5')
    
    parser = argparse.ArgumentParser(
        description="Build RCCL on multiple servers via SSH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Description:
  This script performs pre-flight checks on local repositories:
  1. Main repository (/home/dn/amd-dev):
     - FAILS if uncommitted changes or unpushed commits touch dn/ directory
     - WARNS (continues) if changes/commits are outside dn/ directory
  2. dn/rccl repository:
     - Checks for uncommitted changes (FAILS if found)
     - Checks for unpushed commits to origin (FAILS if found)
  3. dn/amd-anp repository:
     - Checks for uncommitted changes (FAILS if found)
     - Checks for unpushed commits to origin (FAILS if found)
  
  Branch defaults are read from branch-list.txt unless explicitly provided via command line.
  
  If checks pass, it connects to each server in the provided list via SSH and:
  1. Fetches latest git changes (git fetch -p)
  2. Rebases with remote (git pull --rebase)
  3. Checks out the specified branch (git checkout <branch-name>)
  4. Runs dn/build.sh with specified RCCL and AMD-ANP branches
  
  All servers are built in parallel using separate threads for faster execution.
  
  Server List:
  - If -s option is not provided, servers are read from server-list.txt
  - The file should contain one server IP/hostname per line
  - Lines starting with '#' are treated as comments and ignored
  
  Branch List:
  - Default branches are read from branch-list.txt
  - Format: repo-name branch-name (one per line)
  - Command line arguments override file defaults

Examples:
  %(prog)s                        # Use all defaults from branch-list.txt
  %(prog)s -b feature/test        # Override main branch only
  %(prog)s -b main --rccl-branch drop/2025-10 --amd-anp-branch tags/v1.2.0
  %(prog)s -s "192.168.1.10,192.168.1.11" -b main --npkit
  %(prog)s --rccl-branch develop --npkit  # Override RCCL branch, use others from file
        """
    )
    
    parser.add_argument(
        "-s", "--servers",
        help="Comma-separated list of server IPs or hostnames (e.g., '192.168.1.10,192.168.1.11'). If not provided, reads from server-list.txt"
    )
    
    parser.add_argument(
        "-b", "--branch",
        default=default_main_branch,
        help=f"Main repository branch to checkout and build (default: {default_main_branch or 'from branch-list.txt'})"
    )
    
    parser.add_argument(
        "--rccl-branch",
        default=default_rccl_branch,
        help=f"RCCL branch/tag to checkout (default: {default_rccl_branch})"
    )
    
    parser.add_argument(
        "--amd-anp-branch",
        default=default_amd_anp_branch,
        help=f"AMD-ANP branch/tag to checkout (default: {default_amd_anp_branch})"
    )
    
    parser.add_argument(
        "--npkit",
        action="store_true",
        help="Enable NPKit profiling support in RCCL build"
    )
    
    args = parser.parse_args()
    
    # Validate that we have a main branch (either from file or command line)
    if not args.branch:
        print_error("No main branch specified and no default found in branch-list.txt")
        print_error("Please specify a branch with -b/--branch or add 'amd-dev' entry to branch-list.txt")
        sys.exit(1)
    
    # Display branch configuration
    print_info("=" * 48)
    print_info("BRANCH CONFIGURATION")
    print_info("=" * 48)
    if branch_defaults:
        print_info(f"Loaded defaults from branch-list.txt:")
        for repo, branch in sorted(branch_defaults.items()):
            print_info(f"  {repo}: {branch}")
        print()
    
    print_info("Branches to be used for build:")
    print_info(f"  Main (amd-dev): {args.branch}")
    print_info(f"  RCCL: {args.rccl_branch}")
    print_info(f"  AMD-ANP: {args.amd_anp_branch}")
    print()
    
    # Check for uncommitted changes in local repository first
    print_info("Checking local repository for uncommitted changes under dn/...")
    if not check_local_git_status():
        print_error("Build aborted due to uncommitted changes under dn/ in local repository.")
        sys.exit(1)
    print_info("Local repository has no uncommitted changes under dn/.")
    print()
    
    # Check for unpushed commits
    if not check_unpushed_commits():
        print_error("Build aborted due to unpushed commits touching dn/ in local repository.")
        sys.exit(1)
    print_info("All commits touching dn/ are pushed to origin.")
    print()
    
    # Check dn/rccl subdirectory
    print_info("Checking dn/rccl repository for uncommitted changes...")
    if not check_subdir_git_status("/home/dn/amd-dev/dn/rccl", "dn/rccl"):
        print_error("Build aborted due to uncommitted changes in dn/rccl repository.")
        sys.exit(1)
    print_info("dn/rccl repository is clean.")
    print()
    
    print_info("Checking dn/rccl repository for unpushed commits...")
    if not check_subdir_unpushed_commits("/home/dn/amd-dev/dn/rccl", "dn/rccl"):
        print_error("Build aborted due to unpushed commits in dn/rccl repository.")
        sys.exit(1)
    print_info("All commits are pushed to origin in dn/rccl.")
    print()
    
    # Check dn/amd-anp subdirectory
    print_info("Checking dn/amd-anp repository for uncommitted changes...")
    if not check_subdir_git_status("/home/dn/amd-dev/dn/amd-anp", "dn/amd-anp"):
        print_error("Build aborted due to uncommitted changes in dn/amd-anp repository.")
        sys.exit(1)
    print_info("dn/amd-anp repository is clean.")
    print()
    
    print_info("Checking dn/amd-anp repository for unpushed commits...")
    if not check_subdir_unpushed_commits("/home/dn/amd-dev/dn/amd-anp", "dn/amd-anp"):
        print_error("Build aborted due to unpushed commits in dn/amd-anp repository.")
        sys.exit(1)
    print_info("All commits are pushed to origin in dn/amd-anp.")
    print()
    
    # Parse server list
    if args.servers:
        # Servers provided via command line
        servers = [s.strip() for s in args.servers.split(",") if s.strip()]
        print_info("Using servers from command line")
    else:
        # Read servers from file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_file = os.path.join(script_dir, "server-list.txt")
        print_info(f"Reading servers from {server_file}")
        servers = read_servers_from_file(server_file)
    
    if not servers:
        print_error("No valid servers provided or found in server-list.txt")
        sys.exit(1)
    
    if args.npkit:
        print_info("NPKit profiling will be enabled")
    
    print_info("Starting cluster build process (parallel execution)")
    print_info(f"Branch: {args.branch}")
    print_info(f"RCCL Branch: {args.rccl_branch}")
    print_info(f"AMD-ANP Branch: {args.amd_anp_branch}")
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
            args=(server, args.branch, args.npkit, args.rccl_branch, args.amd_anp_branch, 
                  success_servers, failed_servers, list_lock),
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

