"""
HuggingFace Model Adapter for TCR-pMHC translation inference.

This module provides a wrapper around HuggingFace models for performing
inference with various decoding strategies.
"""
import re
import warnings
from typing import Dict, List, Optional, Union

import einops
import torch
import torch.nn.functional as F
from transformers import BartTokenizer, T5Tokenizer, LogitsProcessor


class HuggingFaceModelAdapter:
    """
    Adapter class for HuggingFace models to perform TCR-pMHC translation.

    This class wraps a HuggingFace model and tokenizer to provide a unified
    interface for various decoding strategies including greedy decoding,
    beam search, top-k sampling, top-p sampling, and more.

    Args:
        hf_tokenizer: A HuggingFace tokenizer instance.
        hf_model: A HuggingFace model instance.
        use_task_prefix (bool): Whether to use task prefixes. Defaults to False.
        device (str): The device to run inference on. Defaults to 'cpu'.
    """

    def __init__(self, hf_tokenizer, hf_model, **kwargs):
        self.tokenizer = hf_tokenizer
        self.model = hf_model
        self.use_task_prefix = kwargs.get("use_task_prefix", False)
        self.device = kwargs.get("device", "cpu")

    def format_input(self, source):
        """
        Prepare the input for the model.

        Args:
            source: A TCR or pMHC object from the TCRpMHC dataset.

        Returns:
            BatchEncoding: The tokenized source input for the model.
        """
        # Source is a pMHC and target is a TCR (pMHC -> TCR)
        if hasattr(source, "peptide"):
            if isinstance(self.tokenizer, BartTokenizer):
                tokenized_src = self.tokenizer(
                    source.peptide, source.pseudo, return_tensors="pt"
                ).to(self.device)
            elif isinstance(self.tokenizer, T5Tokenizer):
                seq = f"[PMHC]{source.peptide}{self.tokenizer.sep_token}{source.pseudo}"
                tokenized_src = self.tokenizer(seq, return_tensors="pt").to(self.device)
            else:
                raise NotImplementedError(
                    "This tokenizer has not been implemented or used in training."
                )
            return tokenized_src

        # Source is a TCR and target is a pMHC (TCR -> pMHC)
        elif hasattr(source, "cdr3b"):
            if isinstance(self.tokenizer, BartTokenizer):
                seq = source.cdr3b
            elif isinstance(self.tokenizer, T5Tokenizer):
                seq = f"[TCR]{source.cdr3b}"
            else:
                raise NotImplementedError(
                    "This tokenizer has not been implemented or used in training."
                )
            tokenized_src = self.tokenizer(seq, return_tensors="pt").to(self.device)
            return tokenized_src
        else:
            raise ValueError(
                "This adapter must be used with a TCRpMHCDataset object yielding TCR and pMHC."
            )

    def format_output(self, trg: str) -> str:
        """
        Format the output of the model to be human readable.

        Args:
            trg: The raw model output string.

        Returns:
            str: The formatted output string with special tokens removed.
        """
        pattern = r"\[.*?\]"
        return re.sub(pattern, "", trg)

    @staticmethod
    def rearrange_logits(model_output, softmax: bool = True) -> torch.Tensor:
        """
        Rearrange HF model output scores (logits) into a more interpretable format.

        Args:
            model_output: Dictionary with model outputs including 'scores'.
            softmax: Whether to apply softmax to the logits.

        Returns:
            torch.Tensor: Rearranged logits of shape [bsz, output_size, vocab_size].
        """
        output_collated_tensors = torch.stack(model_output.scores)
        output_collated_tensors = einops.rearrange(
            output_collated_tensors, "seq_len bsz vocab_size -> bsz seq_len vocab_size"
        )
        if softmax:
            return F.softmax(output_collated_tensors, dim=-1)
        return output_collated_tensors

    @staticmethod
    def rearrange_xattn(model_output) -> torch.Tensor:
        """
        Rearrange HF model cross attention output.

        Args:
            model_output: Dictionary with model outputs including 'cross_attentions'.

        Returns:
            torch.Tensor: Cross attention of shape
                [num_decoders, bsz, num_attn_heads, output_size, input_size].
        """
        decoder_collated_tensors = [
            torch.stack(model_output.cross_attentions[i])
            for i in range(len(model_output.cross_attentions))
        ]
        x_attn = torch.cat(decoder_collated_tensors, dim=3)
        return x_attn

    def _greedy_decoding(
        self, tokenized_input, max_len: int, n: int = 1, return_dict: bool = False
    ):
        """
        Implements greedy decoding (deterministic, argmax over auto-regressively
        sampled logits).

        Args:
            tokenized_input: The tokenized input.
            max_len: The maximum length of the generated sequence.
            n: The number of sequences to return (must be 1 for greedy).
            return_dict: Whether to return the dictionary of outputs.

        Returns:
            Model output sequences or dictionary.
        """
        return self.model.generate(
            **tokenized_input,
            max_new_tokens=max_len,
            do_sample=False,
            num_beams=1,
            num_return_sequences=n,
            return_dict_in_generate=return_dict,
        )

    def _multinomial_sampling(
        self,
        tokenized_input,
        max_len: int,
        n: int,
        temperature: float,
        return_dict: bool = False,
    ):
        """
        Implements multinomial (ancestral) sampling from the logits distribution.

        Args:
            tokenized_input: The tokenized input.
            max_len: The maximum length of the generated sequence.
            n: The number of sequences to return.
            temperature: The temperature of the softmax function.
            return_dict: Whether to return the dictionary of outputs.

        Returns:
            Model output sequences or dictionary.
        """
        return self.model.generate(
            **tokenized_input,
            max_new_tokens=max_len,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n,
            return_dict_in_generate=return_dict,
        )

    def _top_k_sampling(
        self,
        tokenized_input,
        max_len: int,
        n: int,
        top_k: int,
        temperature: float,
        return_dict: bool = False,
    ):
        """
        Implements top-k sampling (stochastic sampling from the top-k logits).

        Args:
            tokenized_input: The tokenized input.
            max_len: The maximum length of the generated sequence.
            n: The number of sequences to return.
            top_k: The number of top tokens to sample from.
            temperature: The temperature of the softmax function.
            return_dict: Whether to return the dictionary of outputs.

        Returns:
            Model output sequences or dictionary.
        """
        return self.model.generate(
            **tokenized_input,
            max_new_tokens=max_len,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=n,
            return_dict_in_generate=return_dict,
        )

    def _top_p_sampling(
        self,
        tokenized_input,
        max_len: int,
        n: int,
        top_p: float,
        temperature: Optional[float] = None,
        return_dict: bool = False,
    ):
        """
        Implements top-p (nucleus) sampling.

        Args:
            tokenized_input: The tokenized input.
            max_len: The maximum length of the generated sequence.
            n: The number of sequences to return.
            top_p: The cumulative probability threshold.
            temperature: The temperature of the softmax function.
            return_dict: Whether to return the dictionary of outputs.

        Returns:
            Model output sequences or dictionary.
        """
        return self.model.generate(
            **tokenized_input,
            max_new_tokens=max_len,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            top_k=0,
            num_return_sequences=n,
            return_dict_in_generate=return_dict,
        )

    def _beam_search(
        self,
        tokenized_input,
        max_len: int,
        n: int,
        num_beams: int,
        no_repeat_ngram_size: int,
        do_sample: bool = False,
        return_dict: bool = False,
    ):
        """
        Implements deterministic and stochastic beam search.

        Args:
            tokenized_input: The tokenized input.
            max_len: The maximum length of the generated sequence.
            n: The number of sequences to return (must be <= num_beams).
            num_beams: The number of beams to use.
            no_repeat_ngram_size: The size of the n-gram to avoid repeating.
            do_sample: Whether to sample from the logits.
            return_dict: Whether to return the dictionary of outputs.

        Returns:
            Model output sequences or dictionary.
        """
        assert n <= num_beams, "num_return_sequences must be <= num_beams"
        return self.model.generate(
            **tokenized_input,
            max_new_tokens=max_len,
            num_return_sequences=n,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            return_dict_in_generate=return_dict,
        )

    def _diverse_beam_search(
        self,
        tokenized_input,
        max_len: int,
        n: int,
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        no_repeat_ngram_size: int,
        return_dict: bool = False,
    ):
        """
        Implements diverse beam search.

        Args:
            tokenized_input: The tokenized input.
            max_len: The maximum length of the generated sequence.
            n: The number of sequences to return.
            num_beams: The number of beams to use.
            num_beam_groups: The number of groups for diverse beam search.
            diversity_penalty: The penalty to apply for diversity.
            no_repeat_ngram_size: The size of the n-gram to avoid repeating.
            return_dict: Whether to return the dictionary of outputs.

        Returns:
            Model output sequences or dictionary.
        """
        assert n <= num_beams, "num_return_sequences must be <= num_beams"
        return self.model.generate(
            **tokenized_input,
            max_new_tokens=max_len,
            num_return_sequences=n,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            no_repeat_ngram_size=no_repeat_ngram_size,
            diversity_penalty=diversity_penalty,
            do_sample=False,
            return_dict_in_generate=return_dict,
        )

    def _contrastive_decoding(
        self,
        tokenized_input,
        max_len: int,
        n: int,
        penalty_alpha: float,
        top_k: int,
        return_dict: bool = False,
    ):
        """
        Implements contrastive decoding.

        Args:
            tokenized_input: The tokenized input.
            max_len: The maximum length of the generated sequence.
            n: The number of sequences to return.
            penalty_alpha: The penalty alpha to apply.
            top_k: The number of top tokens to sample from.
            return_dict: Whether to return the dictionary of outputs.

        Returns:
            Model output sequences or dictionary.
        """
        return self.model.generate(
            **tokenized_input,
            max_new_tokens=max_len,
            num_return_sequences=n,
            top_k=top_k,
            penalty_alpha=penalty_alpha,
            do_sample=True,
            return_dict_in_generate=return_dict,
        )

    def _typical_sampling(
        self,
        tokenized_input,
        max_len: int,
        n: int,
        typical_mass: float,
        min_tokens_to_keep: int,
        return_dict: bool = False,
    ):
        """
        Implements typical sampling.

        Args:
            tokenized_input: The tokenized input.
            max_len: The maximum length of the generated sequence.
            n: The number of sequences to return.
            typical_mass: The typical mass to apply.
            min_tokens_to_keep: The minimum tokens to keep.
            return_dict: Whether to return the dictionary of outputs.

        Returns:
            Model output sequences or dictionary.
        """
        logits_processor = [
            TypicalLogitsProcessor(
                mass=typical_mass,
                filter_value=-float("Inf"),
                min_tokens_to_keep=min_tokens_to_keep,
            )
        ]
        return self.model.generate(
            **tokenized_input,
            max_new_tokens=max_len,
            num_return_sequences=n,
            do_sample=True,
            logits_processor=logits_processor,
            return_dict_in_generate=return_dict,
        )

    def translate(
        self,
        source,
        max_len: int = 25,
        n: int = 1,
        mode: str = "greedy",
        return_logits: bool = False,
        return_xattn: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Translate source to target using various decoding strategies.

        Args:
            source: The source input (TCR or pMHC object).
            max_len: The maximum length of the generated sequence.
            n: The number of sequences to return.
            mode: The decoding strategy. Options:
                - 'greedy': Greedy decoding
                - 'ancestral': Multinomial sampling
                - 'top_k': Top-k sampling
                - 'top_p': Top-p (nucleus) sampling
                - 'beam': Deterministic beam search
                - 'stochastic_beam': Stochastic beam search
                - 'diverse_beam': Diverse beam search
                - 'contrastive': Contrastive decoding
                - 'typical': Typical sampling
            return_logits: Whether to return the logits of the output.
            return_xattn: Whether to return the cross attention.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Dict containing:
                - 'translations': List of generated translations
                - 'cross_attentions': Cross attention tensors (if return_xattn=True)
                - 'logits': Output logits (if return_logits=True)
        """
        self.model.eval()
        tokenized_src = self.format_input(source)

        # Extract kwargs
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", None)
        top_p = kwargs.get("top_p", None)
        num_beams = kwargs.get("num_beams", None)
        no_repeat_ngram_size = kwargs.get("no_repeat_ngram_size", 5)
        num_beam_groups = kwargs.get("num_beam_groups", None)
        diversity_penalty = kwargs.get("diversity_penalty", 1.0)
        penalty_alpha = kwargs.get("penalty_alpha", None)
        typical_mass = kwargs.get("typical_mass", None)
        min_tokens_to_keep = kwargs.get("min_tokens_to_keep", 1)
        skip_special_tokens = kwargs.get("skip_special_tokens", True)

        return_dict = True if (return_xattn or return_logits) else False
        if return_dict and n > 1:
            warnings.warn(
                "Returning logits for n > 1 is not recommended due to memory constraints."
            )
            return_dict = False

        if mode == "greedy":
            outputs = self._greedy_decoding(
                tokenized_src, max_len=max_len, n=n, return_dict=return_dict
            )
        elif mode == "ancestral":
            outputs = self._multinomial_sampling(
                tokenized_src,
                max_len=max_len,
                n=n,
                temperature=temperature,
                return_dict=return_dict,
            )
        elif mode == "top_k":
            assert top_k is not None, "top_k must be specified for top_k mode"
            outputs = self._top_k_sampling(
                tokenized_src,
                max_len=max_len,
                n=n,
                top_k=top_k,
                temperature=temperature,
                return_dict=return_dict,
            )
        elif mode == "top_p":
            assert top_p is not None, "top_p must be specified for top_p mode"
            outputs = self._top_p_sampling(
                tokenized_src,
                max_len=max_len,
                n=n,
                top_p=top_p,
                temperature=temperature,
                return_dict=return_dict,
            )
        elif mode == "beam":
            assert num_beams is not None, "num_beams must be specified for beam mode"
            outputs = self._beam_search(
                tokenized_src,
                max_len=max_len,
                n=n,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                return_dict=return_dict,
            )
        elif mode == "stochastic_beam":
            assert num_beams is not None, "num_beams must be specified"
            outputs = self._beam_search(
                tokenized_src,
                max_len=max_len,
                n=n,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=True,
                return_dict=return_dict,
            )
        elif mode == "diverse_beam":
            assert num_beams is not None, "num_beams must be specified"
            assert num_beam_groups is not None, "num_beam_groups must be specified"
            outputs = self._diverse_beam_search(
                tokenized_src,
                max_len=max_len,
                n=n,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                diversity_penalty=diversity_penalty,
                num_beam_groups=num_beam_groups,
                return_dict=return_dict,
            )
        elif mode == "contrastive":
            assert penalty_alpha is not None, "penalty_alpha must be specified"
            assert top_k is not None, "top_k must be specified"
            outputs = self._contrastive_decoding(
                tokenized_src,
                max_len=max_len,
                n=n,
                penalty_alpha=penalty_alpha,
                top_k=top_k,
                return_dict=return_dict,
            )
        elif mode == "typical":
            assert typical_mass is not None, "typical_mass must be specified"
            outputs = self._typical_sampling(
                tokenized_src,
                max_len=max_len,
                n=n,
                typical_mass=typical_mass,
                min_tokens_to_keep=min_tokens_to_keep,
                return_dict=return_dict,
            )
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

        if not (return_xattn or return_logits):
            translations = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=skip_special_tokens
            )
        else:
            translations = self.tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=skip_special_tokens
            )

        outs = {"translations": [self.format_output(t) for t in translations]}
        if return_xattn:
            outs["cross_attentions"] = self.rearrange_xattn(outputs)
        if return_logits:
            outs["logits"] = self.rearrange_logits(outputs)
        return outs

    def translate_plus(self, source, mode: str = "greedy", max_len: int = 25, **kwargs) -> Dict:
        """
        Translate with logits and cross attention output.

        Args:
            source: The source input (TCR or pMHC object).
            mode: The decoding strategy.
            max_len: The maximum length of the generated sequence.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict with translations, logits, and cross attentions.
        """
        return self.translate(
            source,
            max_len=max_len,
            n=1,
            mode=mode,
            return_logits=True,
            return_xattn=True,
            **kwargs,
        )

    def sample_translations(
        self,
        source,
        max_len: int = 25,
        n: int = 5,
        mode: str = "greedy",
        **kwargs,
    ) -> List[str]:
        """
        Sample n translations from the model.

        Args:
            source: The input source data.
            max_len: The maximum length of the generated translations.
            n: The number of translations to generate.
            mode: The mode of generation.
            **kwargs: Additional keyword arguments.

        Returns:
            List[str]: A list of generated translations.
        """
        outs = self.translate(
            source=source,
            max_len=max_len,
            n=n,
            mode=mode,
            return_logits=False,
            return_xattn=False,
            **kwargs,
        )
        return outs["translations"]


class TypicalLogitsProcessor(LogitsProcessor):
    """
    Typical sampling logits warper.

    Implements typical sampling as described in the literature.
    """

    def __init__(
        self,
        mass: float = 0.9,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # Shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(
            1, last_ind.view(-1, 1)
        )
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
