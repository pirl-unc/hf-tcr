"""
hf-tcr: HuggingFace-based inference and evaluation library for TCR-pMHC translation models.

This library provides tools for:
- Running inference with trained translation models
- Evaluating model performance with various metrics (BLEU, precision, recall, etc.)

Data structures (TCR, pMHC, TCRpMHCdataset) are provided by the tcrpmhcdataset package.
"""

__version__ = "0.1.0"

# Re-export core data structures from tcrpmhcdataset
from tcrpmhcdataset import TCR, pMHC, TCRpMHCdataset

# Tokenizers
from .tokenizer import TCRT5Tokenizer, TCRBartTokenizer

# Model adapter for inference
from .adapter import HuggingFaceModelAdapter, TypicalLogitsProcessor

# Evaluation
from .evaluation import ModelEvaluator

# Constants
from .constants import (
    AA_VOCABULARY,
    PEPTIDE_MAX_LEN,
    MHC_MAX_LEN,
    PSEUDO_MAX_LEN,
    TCR_MAX_LEN,
    CDR3_MAX_LEN,
)

__all__ = [
    # Version
    "__version__",
    # Data structures (from tcrpmhcdataset)
    "TCR",
    "pMHC",
    "TCRpMHCdataset",
    # Tokenizers
    "TCRT5Tokenizer",
    "TCRBartTokenizer",
    # Inference
    "HuggingFaceModelAdapter",
    "TypicalLogitsWarper",
    # Evaluation
    "ModelEvaluator",
    # Constants
    "AA_VOCABULARY",
    "PEPTIDE_MAX_LEN",
    "MHC_MAX_LEN",
    "PSEUDO_MAX_LEN",
    "TCR_MAX_LEN",
    "CDR3_MAX_LEN",
]
