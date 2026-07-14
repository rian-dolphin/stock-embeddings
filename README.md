# Learning Embedding Representations for Stocks from Returns Time Series

> Code originally written in 2021/2022 and refactored to use uv and polars in 2026.

Code for learning distributed representations (embeddings) of financial assets from returns time series, relating to the paper *Stock Embeddings: Learning Distributed Representations for Financial Assets* [[pdf](https://arxiv.org/pdf/2202.08968.pdf)].

## Where to start

**Run [`notebooks/pointwise_embeddings.ipynb`](notebooks/pointwise_embeddings.ipynb).** It is the notebook for the original Stock Embeddings paper and walks the whole pipeline end-to-end: load returns, build target/context sets, train embeddings, and evaluate them on sector classification. After `uv sync` and dropping the data files into `Data/` (see [Data](#data)), it runs top to bottom in a few minutes.

Each notebook maps to a paper:

| Notebook | Paper |
|---|---|
| [`notebooks/pointwise_embeddings.ipynb`](notebooks/pointwise_embeddings.ipynb) | Original Stock Embeddings paper [[pdf](https://arxiv.org/pdf/2202.08968.pdf)] — **start here** |
| [`notebooks/window_embeddings.ipynb`](notebooks/window_embeddings.ipynb) | Window-based variant of the original approach |
| [`notebooks/contrastive/`](notebooks/contrastive) | Contrastive learning for asset embeddings [[pdf](https://arxiv.org/pdf/2407.18645)], [accepted](https://dl.acm.org/doi/10.1145/3677052.3698610) at ICAIF 2024 — the four notebooks build up from a triplet margin loss to the full experiment |
| [`notebooks/ucr/`](notebooks/ucr) | Unpublished experiments transferring the embedding methods to the [UCR/UEA](https://www.timeseriesclassification.com/) time-series-classification benchmark (matrix-factorization embeddings vs Catch22/Rocket baselines). Needs the `ucr` extra (see [Setup](#setup)); datasets download automatically, no `Data/` files required |

A related third paper, leveraging financial news for multimodal asset embeddings [[pdf](https://arxiv.org/pdf/2211.06378)], does not have code in this repo.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). To create the environment and install the `stock_embeddings` package with all dependencies:

```bash
uv sync
```

The UCR benchmark experiments need extra dependencies (sktime, numba, dtaidistance) that the core package doesn't require:

```bash
uv sync --extra ucr
```

Run the test suite:

```bash
uv run pytest
```

The notebooks import the installed package, e.g.:

```python
from stock_embeddings import ReturnsData, ClassificationEmbeddings
```

## Data

Data files are too large to be stored in the repo but feel free to reach out to me via email: dolphrian@gmail.com

The code expects two files in `Data/`:
- `Data/returns_df_611.csv` — daily returns matrix (date index, one column per ticker)
- `Data/historical_stocks.csv` — ticker metadata (`ticker`, `name`, `sector`, `industry`)

The first 10 rows of each data file can be found in the sample files in the `Data/` directory to show the required format.

`ReturnsData` locates these files by searching upward from the current working directory, so the notebooks run regardless of whether Jupyter is launched from the repo root or the `notebooks/` directory.

## Cite as

```
@article{dolphin2022stock,
  title={Stock embeddings: Learning distributed representations for financial assets},
  author={Dolphin, Rian and Smyth, Barry and Dong, Ruihai},
  journal={arXiv preprint arXiv:2202.08968},
  year={2022}
}
```
