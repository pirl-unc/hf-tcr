"""Test that all package imports work correctly."""
import pytest


def test_import_package():
    """Test that the main package can be imported."""
    import hf_tcr
    assert hasattr(hf_tcr, "__version__")


def test_import_dataclasses():
    """Test that TCR and pMHC dataclasses can be imported."""
    from hf_tcr import TCR, pMHC
    assert TCR is not None
    assert pMHC is not None


def test_import_dataset():
    """Test that TCRpMHCdataset can be imported."""
    from hf_tcr import TCRpMHCdataset
    assert TCRpMHCdataset is not None


def test_import_tokenizers():
    """Test that tokenizers can be imported."""
    from hf_tcr import TCRT5Tokenizer, TCRBartTokenizer
    assert TCRT5Tokenizer is not None
    assert TCRBartTokenizer is not None


def test_import_adapter():
    """Test that HuggingFaceModelAdapter can be imported."""
    from hf_tcr import HuggingFaceModelAdapter
    assert HuggingFaceModelAdapter is not None


def test_import_evaluator():
    """Test that ModelEvaluator can be imported."""
    from hf_tcr import ModelEvaluator
    assert ModelEvaluator is not None


def test_import_constants():
    """Test that constants can be imported."""
    from hf_tcr import (
        AA_VOCABULARY,
        PEPTIDE_MAX_LEN,
        CDR3_MAX_LEN,
    )
    assert len(AA_VOCABULARY) == 20
    assert PEPTIDE_MAX_LEN == 15
    assert CDR3_MAX_LEN == 23


def test_import_hla_maps_from_tcrpmhcdataset():
    """Test that HLA maps can be imported from tcrpmhcdataset."""
    from tcrpmhcdataset import HLA_SEQUENCE_MAP, HLA_PSEUDO_MAP
    assert isinstance(HLA_SEQUENCE_MAP, dict)
    assert isinstance(HLA_PSEUDO_MAP, dict)
