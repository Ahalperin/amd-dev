#!/usr/bin/env python3
"""
Database operations for storing and retrieving RCCL optimization results.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class ResultsDatabase:
    """Manages storage and retrieval of optimization results."""
    
    def __init__(self, db_path: str):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                test_name TEXT NOT NULL,
                parameters TEXT NOT NULL,
                env_vars TEXT NOT NULL,
                command TEXT NOT NULL,
                
                -- Performance metrics
                busbw_oop REAL,
                busbw_ip REAL,
                algbw_oop REAL,
                algbw_ip REAL,
                time_oop REAL,
                time_ip REAL,
                
                -- Test configuration
                message_size INTEGER,
                num_processes INTEGER,
                
                -- Execution info
                status TEXT NOT NULL,
                return_code INTEGER,
                error_message TEXT,
                execution_time REAL,
                
                -- Raw output
                stdout TEXT,
                stderr TEXT
            )
        """)
        
        # Optimization session metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                config TEXT NOT NULL,
                method TEXT NOT NULL,
                objective TEXT NOT NULL,
                best_value REAL,
                best_run_id INTEGER,
                total_iterations INTEGER,
                status TEXT NOT NULL,
                FOREIGN KEY (best_run_id) REFERENCES optimization_runs(id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON optimization_runs(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status 
            ON optimization_runs(status)
        """)
        
        self.conn.commit()
    
    def insert_run(self, run_data: Dict[str, Any]) -> int:
        """Insert a new optimization run.
        
        Args:
            run_data: Dictionary containing run information
            
        Returns:
            ID of inserted row
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO optimization_runs (
                timestamp, test_name, parameters, env_vars, command,
                busbw_oop, busbw_ip, algbw_oop, algbw_ip, time_oop, time_ip,
                message_size, num_processes,
                status, return_code, error_message, execution_time,
                stdout, stderr
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_data.get('timestamp', datetime.now().isoformat()),
            run_data.get('test_name'),
            json.dumps(run_data.get('parameters', {})),
            json.dumps(run_data.get('env_vars', {})),
            run_data.get('command'),
            run_data.get('busbw_oop'),
            run_data.get('busbw_ip'),
            run_data.get('algbw_oop'),
            run_data.get('algbw_ip'),
            run_data.get('time_oop'),
            run_data.get('time_ip'),
            run_data.get('message_size'),
            run_data.get('num_processes'),
            run_data.get('status'),
            run_data.get('return_code'),
            run_data.get('error_message'),
            run_data.get('execution_time'),
            run_data.get('stdout'),
            run_data.get('stderr')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def create_session(self, config: Dict[str, Any], method: str, objective: str) -> int:
        """Create a new optimization session.
        
        Args:
            config: Configuration dictionary
            method: Optimization method
            objective: Objective metric
            
        Returns:
            Session ID
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO optimization_sessions (
                start_time, config, method, objective, status, total_iterations
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            json.dumps(config),
            method,
            objective,
            'running',
            0
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def update_session(self, session_id: int, **kwargs):
        """Update session information.
        
        Args:
            session_id: Session ID
            **kwargs: Fields to update
        """
        cursor = self.conn.cursor()
        
        updates = []
        values = []
        for key, value in kwargs.items():
            updates.append(f"{key} = ?")
            values.append(value)
        
        values.append(session_id)
        
        query = f"""
            UPDATE optimization_sessions 
            SET {', '.join(updates)}
            WHERE id = ?
        """
        
        cursor.execute(query, values)
        self.conn.commit()
    
    def get_successful_runs(self, session_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all successful optimization runs.
        
        Args:
            session_id: Optional session ID to filter by
            
        Returns:
            List of run dictionaries
        """
        cursor = self.conn.cursor()
        
        query = """
            SELECT * FROM optimization_runs 
            WHERE status = 'success' AND busbw_oop IS NOT NULL
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        runs = []
        for row in rows:
            run = dict(row)
            run['parameters'] = json.loads(run['parameters'])
            run['env_vars'] = json.loads(run['env_vars'])
            runs.append(run)
        
        return runs
    
    def get_best_run(self, objective: str = 'busbw_oop') -> Optional[Dict[str, Any]]:
        """Get the best run based on objective metric.
        
        Args:
            objective: Metric to optimize
            
        Returns:
            Best run dictionary or None
        """
        cursor = self.conn.cursor()
        
        query = f"""
            SELECT * FROM optimization_runs 
            WHERE status = 'success' AND {objective} IS NOT NULL
            ORDER BY {objective} DESC
            LIMIT 1
        """
        
        cursor.execute(query)
        row = cursor.fetchone()
        
        if row:
            run = dict(row)
            run['parameters'] = json.loads(run['parameters'])
            run['env_vars'] = json.loads(run['env_vars'])
            return run
        
        return None
    
    def get_run_count(self) -> int:
        """Get total number of runs.
        
        Returns:
            Number of runs
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM optimization_runs")
        return cursor.fetchone()[0]
    
    def get_successful_run_count(self) -> int:
        """Get number of successful runs.
        
        Returns:
            Number of successful runs
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM optimization_runs 
            WHERE status = 'success' AND busbw_oop IS NOT NULL
        """)
        return cursor.fetchone()[0]
    
    def export_to_csv(self, output_path: str, objective: str = 'busbw_oop'):
        """Export results to CSV file.
        
        Args:
            output_path: Path to output CSV file
            objective: Metric to include
        """
        import pandas as pd
        
        runs = self.get_successful_runs()
        
        if not runs:
            print("No successful runs to export")
            return
        
        # Flatten parameters into columns
        rows = []
        for run in runs:
            row = {
                'timestamp': run['timestamp'],
                'busbw_oop': run['busbw_oop'],
                'busbw_ip': run['busbw_ip'],
                'algbw_oop': run['algbw_oop'],
                'algbw_ip': run['algbw_ip'],
            }
            # Add parameters as columns
            for param, value in run['parameters'].items():
                row[f'param_{param}'] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(rows)} runs to {output_path}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


