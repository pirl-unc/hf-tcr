"""
Constants for TCR-Translate tokenizers.

Note: Biological constants (HLA_SEQUENCE_MAP, HLA_PSEUDO_MAP, etc.) are
available from the tcrpmhcdataset package.
"""
from pathlib import Path

# Package directory paths
PACKAGE_DIR = Path(__file__).parent.resolve()
ASSETS_DIR = PACKAGE_DIR / "assets"

# Sequence length constants for tokenization
PEPTIDE_MAX_LEN = 15
MHC_MAX_LEN = 365
PSEUDO_MAX_LEN = 34
TCR_MAX_LEN = 325
CDR3_MAX_LEN = 23

# Amino acid vocabulary
AA_VOCABULARY = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
]
