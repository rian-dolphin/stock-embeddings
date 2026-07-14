# Learning Embedding Representations for Stocks from Returns Time Series

> Code originally written in 2021/2022 and refactored to use uv and polars in 2026.

Code for learning distributed representations (embeddings) of financial assets from returns time series, relating to the paper *Stock Embeddings: Learning Distributed Representations for Financial Assets* [[pdf](https://arxiv.org/pdf/2202.08968.pdf)].

Relevant papers:
- Original work on learning Stock Embeddings [[pdf](https://arxiv.org/pdf/2202.08968.pdf)]
- Improved results using contrastive learning for asset embeddings [[pdf](https://arxiv.org/pdf/2407.18645)]
  - [Accepted](https://dl.acm.org/doi/10.1145/3677052.3698610) at ICAIF 2024
- Leveraging financial news for multimodal asset embeddings [[pdf](https://arxiv.org/pdf/2211.06378)]

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). To create the environment and install the `stock_embeddings` package with all dependencies:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

The notebooks import the installed package, e.g.:

```python
from stock_embeddings import ReturnsData, ClassificationEmbeddings
```

Further experiments (UCR benchmarks, matrix-factorization baselines, hedging analysis, thesis plots) are archived on the [`working`](../../tree/working) branch.

## Data

Data files are too large to be stored in the repo but feel free to reach out to me via email: dolphrian@gmail.com

The code expects two files in `Data/`:
- `Data/returns_df_611.csv` — daily returns matrix (date index, one column per ticker)
- `Data/historical_stocks.csv` — ticker metadata (`ticker`, `name`, `sector`, `industry`)

The first 10 rows of each data file can be found in the sample files in the `Data/` directory to show the required format.

## Cite as

```
@article{dolphin2022stock,
  title={Stock embeddings: Learning distributed representations for financial assets},
  author={Dolphin, Rian and Smyth, Barry and Dong, Ruihai},
  journal={arXiv preprint arXiv:2202.08968},
  year={2022}
}
```
