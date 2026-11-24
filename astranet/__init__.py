"""
ASTRA — Automated Spectral TRansient Analysis

Hierarchical classifier for astronomical transient spectra.
"""

from .model   import ASTRANet, build_hierarchy_mappings
from .config  import ASTRAConfig
from .dataset import SpectraDataset, build_loaders

__version__ = "1.0.0"
__all__ = ["ASTRANet", "ASTRAConfig", "SpectraDataset", "build_loaders",
           "build_hierarchy_mappings"]
