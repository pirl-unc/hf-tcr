"""
Model evaluation metrics for TCR-pMHC translation.

This module provides comprehensive evaluation functionality for measuring
the performance of translation models on TCR-pMHC datasets.
"""
import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import Levenshtein as levenshtein
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

from torch.utils.data import DataLoader
from tcrpmhcdataset import TCRpMHCdataset

from .adapter import HuggingFaceModelAdapter
from .tokenizer import TCRT5Tokenizer, TCRBartTokenizer


def _create_collate_fn(tokenizer, dataset: TCRpMHCdataset):
    """
    Create a collate function that uses the proper tokenization methods.

    This ensures tokenization is consistent with training by using
    tokenize_tcr and tokenize_pmhc methods for our custom tokenizers.

    Args:
        tokenizer: A tokenizer instance (TCRT5Tokenizer or TCRBartTokenizer).
        dataset: A TCRpMHCdataset instance to determine source/target direction.

    Returns:
        A collate function for use with DataLoader.
    """
    def _collate_fn(batch):
        source_batch, target_batch = tuple(zip(*batch))

        # Determine which is pMHC and which is TCR based on dataset.source
        if dataset.source == 'pmhc':
            pmhc_batch = source_batch
            tcr_batch = target_batch
        else:
            pmhc_batch = target_batch
            tcr_batch = source_batch

        # Extract sequences for tokenization
        tcr_sequences = [tcr.cdr3b for tcr in tcr_batch]
        pmhc_tuples = [(pmhc.peptide, pmhc.pseudo) for pmhc in pmhc_batch]

        # Use the proper tokenization methods
        if hasattr(tokenizer, 'tokenize_tcr') and hasattr(tokenizer, 'tokenize_pmhc'):
            batched_tcrs = tokenizer.tokenize_tcr(
                tcr_sequences,
                padding=True,
                truncation=True,
                max_length=25,
                return_tensors='pt'
            )
            batched_pmhcs = tokenizer.tokenize_pmhc(
                pmhc_tuples,
                padding=True,
                truncation=True,
                max_length=52,
                return_tensors='pt'
            )
        else:
            # Fallback for generic tokenizers (without task prefixes)
            batched_tcrs = tokenizer(
                tcr_sequences,
                padding=True,
                truncation=True,
                max_length=25,
                return_tensors='pt'
            )
            pmhc_strings = [f"{pep} {pseudo}" for pep, pseudo in pmhc_tuples]
            batched_pmhcs = tokenizer(
                pmhc_strings,
                padding=True,
                truncation=True,
                max_length=52,
                return_tensors='pt'
            )

        # Return batched data with labels based on translation direction
        if dataset.source == 'pmhc':
            batched_pmhcs['labels'] = batched_tcrs['input_ids']
            return batched_pmhcs
        else:
            batched_tcrs['labels'] = batched_pmhcs['input_ids']
            return batched_tcrs

    return _collate_fn


def _get_dataloader(dataset: TCRpMHCdataset, tokenizer, batch_size: int = 1,
                    shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader with proper tokenization for TCR-Translate models.

    Args:
        dataset: A TCRpMHCdataset instance.
        tokenizer: A tokenizer instance.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes.

    Returns:
        DataLoader with custom collate function.
    """
    collate_fn = _create_collate_fn(tokenizer, dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def _dataset_to_seq2seq_dict(dataset: TCRpMHCdataset, stringify_keys: bool = False) -> Dict:
    """
    Create a source-to-target dictionary from a TCRpMHCdataset.

    Args:
        dataset: A TCRpMHCdataset instance.
        stringify_keys: If True, use string representations as keys.
                       If False, use original objects as keys.

    Returns:
        Dictionary mapping source (objects or strings) to lists of target strings.
    """
    src_list = dataset.get_srclist()
    trg_list = dataset.get_trglist()

    seq2seq_dict = {}
    for src, trg in zip(src_list, trg_list):
        # Use object or string as key based on stringify_keys
        key = str(src) if stringify_keys else src
        trg_str = str(trg) if not isinstance(trg, str) else trg

        if key not in seq2seq_dict:
            seq2seq_dict[key] = []
        if trg_str not in seq2seq_dict[key]:
            seq2seq_dict[key].append(trg_str)

    return seq2seq_dict


class ModelEvaluator(HuggingFaceModelAdapter):
    """
    Evaluator class for measuring translation model performance.

    This class extends HuggingFaceModelAdapter with methods for computing
    various evaluation metrics including BLEU, precision, recall, F1,
    edit distance, and sequence recovery.

    Args:
        hf_tokenizer: A HuggingFace tokenizer instance.
        hf_model: A HuggingFace model instance.
        **kwargs: Additional arguments passed to HuggingFaceModelAdapter.
    """

    def __init__(self, hf_tokenizer, hf_model, **kwargs):
        super().__init__(hf_tokenizer=hf_tokenizer, hf_model=hf_model, **kwargs)

    @staticmethod
    def find_n_closest_matches(
        query: str, references: List[str], n: int
    ) -> List[str]:
        """
        Find the n closest matches to the query in the reference list.

        Args:
            query: The query string to match.
            references: List of reference strings.
            n: Number of closest matches to return.

        Returns:
            List of n closest matching strings.
        """
        distances = [(ref, levenshtein.distance(query, ref)) for ref in references]
        distances.sort(key=lambda x: x[1])
        return [d[0] for d in distances[:n]]

    @staticmethod
    def _sequence_bleu(
        translation: str,
        references: List[str],
        max_references: int = 20,
        max_ngram: int = 4,
    ) -> float:
        """
        Calculate the sequence level Char-BLEU score for a single TCR or pMHC.

        Uses the NLTK Sentence-Bleu function with character-level tokenization.
        The BLEU score is calculated by treating each character as a word.

        Args:
            translation: The hypothesis translation to be compared.
            references: A list of reference translations.
            max_references: Maximum number of reference translations to consider.
            max_ngram: The maximum n-gram to consider. Defaults to BLEU-4.

        Returns:
            float: The sequence-level Char-BLEU score.
        """
        references = [
            list(x)
            for x in ModelEvaluator.find_n_closest_matches(
                translation, references, n=max_references
            )
        ]
        translation = list(translation)
        return float(
            sentence_bleu(
                references,
                translation,
                weights=tuple([1 / max_ngram] * max_ngram),
                smoothing_function=None,
            )
        )

    @staticmethod
    def _dataset_bleu(
        translations: List[str],
        references: List[List[str]],
        max_references: int = 20,
        max_ngram: int = 4,
        verbose: bool = False,
    ) -> float:
        """
        Calculate the BLEU score for a list of hypotheses and references.

        Args:
            translations: A list of generated translations.
            references: A list of lists of reference translations.
            max_references: Maximum number of references to consider per translation.
            max_ngram: The maximum n-gram to consider. Defaults to BLEU-4.
            verbose: Whether to print debug information.

        Returns:
            float: The corpus-level Char-BLEU score.
        """
        chencherry = SmoothingFunction()

        expanded_references = []
        expanded_translations = []

        for idx, translation in enumerate(translations):
            expanded_references.append(
                [
                    list(x)
                    for x in ModelEvaluator.find_n_closest_matches(
                        translation, references[idx], n=max_references
                    )
                ]
            )
            expanded_translations.append(list(translation))

        if verbose:
            print(f"Expanded References:{expanded_references}")
            print(f"Expanded Translations:{expanded_translations}")

        return float(
            corpus_bleu(
                expanded_references,
                expanded_translations,
                weights=tuple([1 / max_ngram] * max_ngram),
                smoothing_function=chencherry.method1,
            )
        )

    def dataset_bleu(
        self,
        dataset: TCRpMHCdataset,
        max_references: int = 20,
        max_len: int = 25,
        max_ngram: int = 4,
    ) -> float:
        """
        Calculate the Dataset level Char-BLEU score for a TCRpMHC dataset.

        Uses greedy decoding to generate translations and compares them
        to reference translations using character-level BLEU.

        Args:
            dataset: A TCRpMHCDataset object.
            max_references: Maximum number of references to consider.
            max_len: Maximum length of generated translations.
            max_ngram: The maximum n-gram to consider. Defaults to BLEU-4.

        Returns:
            float: The character level BLEU score.
        """
        translations = []
        references = []

        seq2seq_mapping = _dataset_to_seq2seq_dict(dataset)
        for src in tqdm(seq2seq_mapping.keys(), desc="Char-BLEU"):
            references.append(seq2seq_mapping[src])
            translations.append(
                self.sample_translations(
                    source=src, n=1, max_len=max_len, mode="greedy"
                )[0]
            )

        return self._dataset_bleu(
            translations, references, max_references=max_references, max_ngram=max_ngram
        )

    @staticmethod
    def _precision_at_k(
        translations: List[str],
        reference_translations: List[str],
        k: Optional[int] = None,
    ) -> float:
        """Calculate precision at k."""
        correct = [t for t in translations if t in reference_translations]
        return len(correct) / len(translations)

    @staticmethod
    def _recall_at_k(
        translations: List[str], reference_translations: List[str], k: int
    ) -> float:
        """Calculate recall at k."""
        correct = [t for t in translations if t in reference_translations]
        return len(set(correct)) / min(k, len(reference_translations))

    @staticmethod
    def _f1_at_k(
        translations: List[str],
        reference_translations: List[str],
        k: Optional[int] = None,
    ) -> float:
        """Calculate F1 score at k."""
        precision = ModelEvaluator._precision_at_k(
            translations, reference_translations, k
        )
        recall = ModelEvaluator._recall_at_k(translations, reference_translations, k)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _mean_edit_distance(
        translations: List[str], reference_translations: List[str]
    ) -> float:
        """Calculate the mean edit distance to closest reference."""
        edit_distances = []
        for translation in translations:
            closest_match = ModelEvaluator.find_n_closest_matches(
                translation, reference_translations, 1
            )[0]
            edit_distances.append(levenshtein.distance(translation, closest_match))
        return sum(edit_distances) / len(translations)

    @staticmethod
    def _mean_sequence_recovery(
        translations: List[str], reference_translations: List[str]
    ) -> float:
        """
        Calculate the mean sequence recovery as a percent for each translation.

        Args:
            translations: List of generated translations.
            reference_translations: List of reference translations.

        Returns:
            float: Mean sequence recovery percentage.
        """
        per_sequence_percents = []
        for translation in translations:
            same_len_references = [
                ref for ref in reference_translations if len(ref) == len(translation)
            ]
            if len(same_len_references) == 0:
                closest_match = ModelEvaluator.find_n_closest_matches(
                    translation, reference_translations, 1
                )[0]
                per_sequence_percents.append(
                    1 - levenshtein.distance(translation, closest_match) / len(closest_match)
                )
                continue
            closest_match = ModelEvaluator.find_n_closest_matches(
                translation, same_len_references, 1
            )[0]
            idx_recovery = [
                1 if char == closest_match[idx] else 0
                for idx, char in enumerate(translation)
            ]
            per_sequence_percents.append(sum(idx_recovery) / len(translation))
        return np.mean(per_sequence_percents)

    def get_batch_size(
        self,
        dataset: TCRpMHCdataset,
        max_memory_usage: float = 0.97,
    ) -> int:
        """
        Determine optimal batch size based on GPU memory.

        Args:
            dataset: The dataset to evaluate.
            max_memory_usage: Maximum fraction of GPU memory to use.

        Returns:
            int: The optimal batch size.
        """
        self.model.eval()
        bsz = 1

        while bsz < min(len(dataset), 2048):
            try:
                if bsz * 2 >= len(dataset):
                    return bsz
                dloader = dataset.get_dataloader(self.tokenizer, batch_size=bsz)

                with torch.no_grad():
                    batch = next(iter(dloader))
                    _ = self.model(**batch.to(self.device))

                memory_allocated = torch.cuda.memory_allocated(self.device)
                memory_cached = torch.cuda.memory_reserved(self.device)
                memory_usage = memory_allocated / (memory_cached + 1)

                if memory_usage <= max_memory_usage:
                    bsz *= 2
                else:
                    break
            except RuntimeError:
                bsz //= 2
                break

        torch.cuda.empty_cache()
        return bsz // 2

    def evaluate_loss(
        self, dataset: TCRpMHCdataset, cumulative: bool = False
    ) -> float:
        """
        Evaluate the loss of the model on a dataset.

        Uses the HuggingFace model loss function (CrossEntropyLoss).

        Args:
            dataset: A TCRpMHCDataset object.
            cumulative: Return cumulative loss if True, else average loss.

        Returns:
            float: The average or cumulative loss across the dataset.
        """
        bsz = min(512, len(dataset))
        if self.device != "cpu":
            bsz = self.get_batch_size(dataset)
        self.model.eval()
        # Use custom dataloader with proper tokenization (tokenize_tcr/tokenize_pmhc)
        dloader = _get_dataloader(dataset, self.tokenizer, batch_size=bsz)
        num_batches = len(dloader)

        cum_loss = 0

        with torch.no_grad():
            for batch in tqdm(dloader, desc="XEntropy Loss"):
                batch.to(self.device)
                outs = self.model(**batch)
                cum_loss += outs.loss
                torch.cuda.empty_cache()

        if cumulative:
            return cum_loss.item()

        return cum_loss.item() / num_batches

    def _precision_recall_f1_at_k(
        self, translations: List[str], ref_trgs: List[str], k: int = 100
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, F1 score at k for generated translations.

        Args:
            translations: A list of generated translations.
            ref_trgs: A list of reference translations.
            k: The number of translations to consider.

        Returns:
            Tuple of (precision, recall, f1).
        """
        precision = self._precision_at_k(translations, ref_trgs, k=k)
        recall = self._recall_at_k(translations, ref_trgs, k=k)
        f1 = self._f1_at_k(translations, ref_trgs, k=k)
        return precision, recall, f1

    def atomic_metrics_at_k(
        self,
        dataset: TCRpMHCdataset,
        k: int = 100,
        max_len: int = 25,
        return_translations: bool = False,
        mode: str = "top_k",
        **kwargs,
    ) -> Dict:
        """
        Calculate performance metrics for each source in the dataset.

        Implements precision, recall, F1 score, edit distance, sequence recovery,
        and BLEU for each source object individually.

        Args:
            dataset: A TCRpMHCDataset object.
            k: The number of translations to consider.
            max_len: Maximum length of generated translations.
            return_translations: Whether to return translations with metrics.
            mode: The mode of generation (e.g., 'top_k', 'greedy').
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Dict mapping source to metrics dict.
        """
        metrics = {
            "char-bleu": -100,
            "precision": -100,
            "recall": -100,
            "f1": -100,
            "d_edit": -100,
            "seq_recovery": -100,
            "translations": None,
            "reference_translations": None,
        }

        reference_dict = _dataset_to_seq2seq_dict(dataset)
        translation_metrics = {}

        for source in tqdm(
            list(set(dataset.get_srclist())), desc="Calculating Atomic Metrics"
        ):
            translation_metrics[source] = copy.deepcopy(metrics)

            src_translations = self.sample_translations(
                source, n=k, max_len=max_len, mode=mode, **kwargs
            )
            trg_sequences = reference_dict[source]

            if return_translations:
                translation_metrics[source]["translations"] = src_translations
                translation_metrics[source]["reference_translations"] = trg_sequences

            translation_metrics[source]["char-bleu"] = self._sequence_bleu(
                self.sample_translations(source, n=1, max_len=max_len, mode="greedy")[0],
                trg_sequences,
                max_ngram=4,
                max_references=20,
            )

            precision, recall, f1 = self._precision_recall_f1_at_k(
                src_translations, trg_sequences, k=k
            )
            translation_metrics[source]["precision"] = precision
            translation_metrics[source]["recall"] = recall
            translation_metrics[source]["f1"] = f1
            translation_metrics[source]["d_edit"] = self._mean_edit_distance(
                src_translations, trg_sequences
            )
            translation_metrics[source]["seq_recovery"] = self._mean_sequence_recovery(
                src_translations, trg_sequences
            )

        return translation_metrics

    def dataset_metrics_at_k(
        self,
        dataset: TCRpMHCdataset,
        k: int = 100,
        max_len: int = 25,
        mode: str = "top_k",
        **kwargs,
    ) -> Dict:
        """
        Calculate performance metrics at dataset-level granularity.

        Args:
            dataset: A TCRpMHCDataset object.
            k: The number of translations to consider.
            max_len: Maximum length of generated translations.
            mode: The mode of generation (e.g., 'top_k', 'greedy').
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Dict containing:
                - char-bleu: Dataset level Char-BLEU score
                - precision: Mean precision score
                - recall: Mean recall score
                - f1: Mean F1 score
                - d_edit: Mean edit distance
                - diversity: Unique sequences / total sequences
                - seq_recovery: Mean sequence recovery
                - perplexity: Perplexity score
        """
        metrics = {
            "char-bleu": -100,
            "precision": [],
            "recall": [],
            "f1": [],
            "d_edit": [],
            "seq_recovery": [],
            "diversity": [],
            "perplexity": -100,
        }

        metrics["char-bleu"] = self.dataset_bleu(dataset)
        metrics["perplexity"] = np.exp(self.evaluate_loss(dataset, cumulative=False))

        reference_dict = _dataset_to_seq2seq_dict(dataset)

        for source in tqdm(
            list(set(dataset.get_srclist())), desc="Calculating @K Metrics"
        ):
            src_translations = self.sample_translations(
                source, n=k, max_len=max_len, mode=mode, **kwargs
            )
            trg_sequences = reference_dict[source]

            precision, recall, f1 = self._precision_recall_f1_at_k(
                src_translations, trg_sequences, k=k
            )
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
            metrics["d_edit"].append(
                self._mean_edit_distance(src_translations, trg_sequences)
            )
            metrics["diversity"] += src_translations
            metrics["seq_recovery"].append(
                self._mean_sequence_recovery(src_translations, trg_sequences)
            )

        metrics["precision"] = np.mean(metrics["precision"])
        metrics["recall"] = np.mean(metrics["recall"])
        metrics["f1"] = np.mean(metrics["f1"])
        metrics["d_edit"] = np.mean(metrics["d_edit"])
        metrics["diversity"] = len(set(metrics["diversity"])) / len(metrics["diversity"])
        metrics["seq_recovery"] = np.mean(metrics["seq_recovery"])

        return metrics

    def stratified_metrics_at_k(
        self,
        dataset: TCRpMHCdataset,
        stratify_on: str = "Allele",
        k: int = 100,
        max_len: int = 25,
        mode: str = "top_k",
        **kwargs,
    ) -> Dict:
        """
        Calculate metrics stratified by a specific column (e.g., Allele).

        Args:
            dataset: A TCRpMHCDataset object.
            stratify_on: The column to stratify on.
            k: The number of translations to consider.
            max_len: Maximum length of generated translations.
            mode: The mode of generation.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Dict mapping group values to their metrics.
        """
        df = dataset.to_df()
        fine_grained_metrics = {}

        groups = df[stratify_on].unique()
        df_list = [
            df[df[stratify_on] == group].reset_index(drop=True) for group in groups
        ]

        dset_list = [
            TCRpMHCdataset(
                source=dataset.source,
                target=dataset.target,
                use_pseudo=dataset.use_pseudo,
                use_cdr3=dataset.use_cdr3,
                use_mhc=dataset.use_mhc,
            )
            for _ in df_list
        ]

        for i, daf in enumerate(df_list):
            dset_list[i].load_data_from_df(daf)

        for i, group in enumerate(groups):
            fine_grained_metrics[group] = self.dataset_metrics_at_k(
                dset_list[i], k=k, max_len=max_len, mode="top_k", **kwargs
            )
            fine_grained_metrics[group]["size"] = (
                len(set(dset_list[i].pMHCs))
                if dataset.source == "pmhc"
                else len(set(dset_list[i].tcrs))
            )

        return fine_grained_metrics
