# hf-tcr

HuggingFace-based inference and evaluation library for TCR-pMHC sequence translation models.

## Installation

```bash
pip install hf-tcr
```

Or install from source:

```bash
git clone https://github.com/pirl-unc/hf-tcr.git
cd hf-tcr
pip install .
```

## Quick Start

### Loading Data

```python
from hf_tcr import TCRpMHCdataset

# Create dataset for pMHC -> TCR translation
dataset = TCRpMHCdataset(
    source="pmhc",
    target="tcr",
    use_pseudo=True,
    use_cdr3=True
)

# Load from CSV file
dataset.load_data_from_file("path/to/data.csv")
```

### Running Inference

```python
from hf_tcr import HuggingFaceModelAdapter, TCRBartTokenizer
from transformers import BartForConditionalGeneration

# Load your trained model and tokenizer
tokenizer = TCRBartTokenizer()
model = BartForConditionalGeneration.from_pretrained("path/to/model")

# Create adapter
adapter = HuggingFaceModelAdapter(
    hf_tokenizer=tokenizer,
    hf_model=model,
    device="cuda"
)

# Get a source from your dataset
source = dataset[0][0]  # Get source from first example

# Generate translations
translations = adapter.sample_translations(
    source=source,
    n=10,
    max_len=25,
    mode="top_k",
    top_k=50,
    temperature=1.0
)
```

### Evaluating Models

```python
from hf_tcr import ModelEvaluator

# Create evaluator (extends HuggingFaceModelAdapter)
evaluator = ModelEvaluator(
    hf_tokenizer=tokenizer,
    hf_model=model,
    device="cuda"
)

# Compute dataset-level metrics
metrics = evaluator.dataset_metrics_at_k(
    dataset=dataset,
    k=100,
    max_len=25,
    mode="top_k",
    top_k=50
)

print(f"BLEU: {metrics['char-bleu']:.4f}")
print(f"Precision@100: {metrics['precision']:.4f}")
print(f"Recall@100: {metrics['recall']:.4f}")
print(f"F1@100: {metrics['f1']:.4f}")
print(f"Mean Edit Distance: {metrics['d_edit']:.2f}")
print(f"Sequence Recovery: {metrics['seq_recovery']:.4f}")
print(f"Diversity: {metrics['diversity']:.4f}")
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

## Available Decoding Strategies

The adapter supports multiple decoding strategies:

- `greedy`: Deterministic greedy decoding
- `ancestral`: Multinomial sampling
- `top_k`: Top-k sampling with temperature
- `top_p`: Nucleus (top-p) sampling
- `beam`: Deterministic beam search
- `stochastic_beam`: Stochastic beam search
- `diverse_beam`: Diverse beam search
- `contrastive`: Contrastive decoding
- `typical`: Typical sampling

## Metrics

The `ModelEvaluator` provides the following metrics:

- **Char-BLEU**: Character-level BLEU score
- **Precision@K**: Fraction of generated sequences that match references
- **Recall@K**: Fraction of reference sequences recovered
- **F1@K**: Harmonic mean of precision and recall
- **Mean Edit Distance**: Average Levenshtein distance to closest reference
- **Sequence Recovery**: Position-wise match percentage
- **Diversity**: Ratio of unique to total generated sequences
- **Perplexity**: Model perplexity on the dataset

## Data Format

CSV files should contain the following columns:

**Required:**
- `CDR3b`: CDR3 beta sequence
- `TRBV`: TRBV gene (IMGT format)
- `TRBJ`: TRBJ gene (IMGT format)
- `Epitope`: Peptide sequence
- `Allele`: HLA allele
- `Reference`: Data source reference

**Optional:**
- `CDR3a`, `TRAV`, `TRAJ`, `TRAD`, `TRBD`
- `TRA_stitched`, `TRB_stitched`
- `Pseudo`, `MHC`

## Dependencies

- torch >= 2.0.0
- transformers >= 4.30.0
- numpy, pandas, tqdm
- python-Levenshtein
- nltk
- einops
- tidytcells >= 2.0.0
- mhcgnomes >= 1.8.0
- tcrpmhcdataset >= 0.2.0

## License

Apache-2.0
