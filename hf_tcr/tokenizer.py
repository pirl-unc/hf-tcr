"""
Custom tokenizers for TCR and pMHC sequences.

This module provides custom wrappers around HuggingFace tokenizers
to handle TCR and pMHC sequence tokenization.
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import T5Tokenizer, BartTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from .constants import ASSETS_DIR

# Define paths to tokenizer files
T5_VOCAB_FILE = str(ASSETS_DIR / "tcrt5_tokenizer.model")
BART_VOCAB_FILE = str(ASSETS_DIR / "tcrbart_vocab.json")
BART_MERGES_FILE = str(ASSETS_DIR / "tcrbart_merges.txt")


class TCRT5Tokenizer(T5Tokenizer):
    """
    Custom T5 tokenizer for TCR and pMHC sequences.

    This tokenizer extends the HuggingFace T5Tokenizer with methods
    for tokenizing TCR and pMHC amino acid sequences.
    """

    def __init__(
        self,
        vocab_file: str = T5_VOCAB_FILE,
        bos_token: str = "[SOS]",
        eos_token: str = "[EOS]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        *args,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            *args,
            **kwargs,
        )

    def tokenize_tcr(
        self, tcr: Union[str, List[str]], **kwargs
    ) -> BatchEncoding:
        """
        Tokenizes TCR amino acid sequence(s).

        Args:
            tcr: List of or single TCR amino acid sequence.
                 Can be full TRB/TRA or CDR3 (Single-chain).
            **kwargs: Additional arguments passed to the tokenizer.

        Returns:
            BatchEncoding: Tokenized TCRs.
        """
        if isinstance(tcr, list):
            tcrs = [f"[TCR]{t}" for t in tcr]
        else:
            tcrs = [f"[TCR]{tcr}"]

        return self.__call__(tcrs, **kwargs)

    def tokenize_pmhc(
        self, pmhc: Union[tuple, List[tuple]], **kwargs
    ) -> BatchEncoding:
        """
        Tokenizes pMHC amino acid sequence tuple(s).

        Args:
            pmhc: List of or single pMHC amino acid sequence tuple.
                  Each tuple should be (peptide, mhc_sequence).
            **kwargs: Additional arguments passed to the tokenizer.

        Returns:
            BatchEncoding: Tokenized pMHCs.
        """
        if isinstance(pmhc, list):
            pmhcs = [
                f"[PMHC]{peptide}{self.sep_token}{mhc}" for (peptide, mhc) in pmhc
            ]
        else:
            pmhcs = [f"[PMHC]{pmhc[0]}{self.sep_token}{pmhc[1]}"]

        return self.__call__(pmhcs, **kwargs)

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor, np.ndarray],
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Decodes token ids to string.

        Args:
            token_ids: List of or single token ids.
            **kwargs: Additional arguments passed to the decoder.

        Returns:
            Decoded string or list of strings.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            raise ValueError(
                "token_ids must be a list, numpy ndarray, or torch tensor."
            )

        if isinstance(token_ids[0], list):
            return [self.decode(t, **kwargs) for t in token_ids]
        else:
            # Remove spaces after [SEP] token
            return super().decode(token_ids, **kwargs).replace(" ", "")


class TCRBartTokenizer(BartTokenizer):
    """
    Custom BART tokenizer for TCR and pMHC sequences.

    This tokenizer extends the HuggingFace BartTokenizer with methods
    for tokenizing TCR and pMHC amino acid sequences.
    """

    def __init__(
        self,
        vocab_file: str = BART_VOCAB_FILE,
        merges_file: str = BART_MERGES_FILE,
        bos_token: str = "[SOS]",
        eos_token: str = "[EOS]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        *args,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            *args,
            **kwargs,
        )

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence
        classification tasks by concatenating and adding special tokens.

        A BART sequence has the following format:
        - single sequence: `[SOS]SEQ1[EOS]`
        - pair of sequences: `[SOS]SEQ1[SEP]SEQ2[EOS]`

        Args:
            token_ids_0: List of IDs for the first sequence.
            token_ids_1: Optional list of IDs for the second sequence.

        Returns:
            List of input IDs with special tokens.
        """
        sos = [self.bos_token_id]
        eos = [self.eos_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return sos + token_ids_0 + eos
        return sos + token_ids_0 + sep + token_ids_1 + eos

    def tokenize_tcr(
        self, tcr: Union[str, List[str]], **kwargs
    ) -> BatchEncoding:
        """
        Tokenizes TCR amino acid sequence(s).

        Args:
            tcr: List of or single TCR amino acid sequence.
                 Can be full TRB/TRA or CDR3 (Single-chain).
            **kwargs: Additional arguments passed to the tokenizer.

        Returns:
            BatchEncoding: Tokenized TCRs.
        """
        if not isinstance(tcr, list):
            tcr = [tcr]

        return self.__call__(tcr, **kwargs)

    def tokenize_pmhc(
        self, pmhc: Union[tuple, List[tuple]], **kwargs
    ) -> BatchEncoding:
        """
        Tokenizes pMHC amino acid sequence tuple(s).

        Args:
            pmhc: List of or single pMHC amino acid sequence tuple.
                  Each tuple should be (peptide, mhc_sequence).
            **kwargs: Additional arguments passed to the tokenizer.

        Returns:
            BatchEncoding: Tokenized pMHCs with format
                "[CLS]PEPTIDE[SEP][SEP]MHCSEQUENCE[SEP]..."
        """
        if not isinstance(pmhc, list):
            pmhc = [pmhc]

        return self.__call__(pmhc, **kwargs)

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor, np.ndarray],
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Decodes token ids to string.

        Args:
            token_ids: List of or single token ids.
            **kwargs: Additional arguments passed to the decoder.

        Returns:
            Decoded string or list of strings.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            raise ValueError(
                "token_ids must be a list, numpy ndarray, or torch tensor."
            )

        if isinstance(token_ids[0], list):
            return [self.decode(t, **kwargs) for t in token_ids]
        else:
            # Remove spaces after [SEP] token
            return super().decode(token_ids, **kwargs).replace(" ", "")
