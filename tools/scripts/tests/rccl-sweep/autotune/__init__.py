"""
RCCL Auto-Tuner Package

Provides automated RCCL performance tuning through:
- Pipeline orchestration
- Hotspot analysis and targeted sweep planning
- Tuner configuration generation
"""

from .pipeline import AutoTunePipeline
from .config_generator import TunerConfigGenerator
from .hotspot_analyzer import HotspotAnalyzer, Hotspot
from .sweep_planner import SweepPlanner, SweepConfig

__all__ = [
    'AutoTunePipeline',
    'TunerConfigGenerator',
    'HotspotAnalyzer',
    'Hotspot',
    'SweepPlanner',
    'SweepConfig',
]

__version__ = '0.1.0'

