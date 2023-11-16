#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:50:56 2021

@author: rian
"""

from enum import Enum

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_plot_from_embeddings(
    embedding_matrix,
    sectors,
    tickers,
    industries,
    names,
    dimensions=2,
    reduced=True,
    method="PCA",
    return_df=False,
    rand_state=None,
):
    """
    Generate a PCA or TSNE plot from embeddings.

    Parameters
    ----------
    embedding_matrix : ndarray
        The embedding matrix to plot.
    sectors, tickers, industries, names : list
        Lists of sectors, tickers, industries, and names for the plot.
    dimensions : int, optional
        Number of dimensions for PCA plot (either 2 or 3). Default is 2.
    reduced : bool, optional
        Whether to plot a subset of sectors. Default is True.
    method : EmbeddingMethod, optional
        Method to use for dimensionality reduction (PCA or TSNE). Default is PCA.
    return_df : bool, optional
        If True, returns the DataFrame used for plotting. Default is False.
    rand_state : int, optional
        Random state for TSNE. Default is None.

    Returns
    -------
    DataFrame or None
        Returns the DataFrame used for plotting if return_df is True.
    """

    if dimensions not in [2, 3]:
        raise ValueError("Dimensions must be either 2 or 3.")

    # Dimensionality reduction
    if method == "PCA":
        reduced_vals = PCA(n_components=3).fit_transform(embedding_matrix)
    elif method == "TSNE":
        reduced_vals = TSNE(n_components=3, random_state=rand_state).fit_transform(
            embedding_matrix
        )
    else:
        raise ValueError("Invalid method. Choose from PCA or TSNE.")

    plot_df = create_dataframe(reduced_vals, sectors, tickers, industries, names)

    # Subset for reduced plot
    if reduced:
        plot_df = plot_df[
            plot_df["sector"].isin(
                ["TECHNOLOGY", "FINANCE", "ENERGY", "PUBLIC UTILITIES"]
            )
        ]

    if dimensions == 2:
        plot_2d(plot_df)
    elif dimensions == 3:
        plot_3d(plot_df)

    if return_df:
        return plot_df


def create_dataframe(vals, sectors, tickers, industries, names):
    """Create a DataFrame for plotting."""
    return pd.DataFrame(
        {
            "latent1": vals[:, 0],
            "latent2": vals[:, 1],
            "latent3": vals[:, 2],
            "sector": sectors,
            "ticker": tickers,
            "industry": industries,
            "name": names,
            "size": [0.5] * len(names),
        }
    )


def plot_2d(df):
    """Plot the 2D scatter plot."""
    fig = px.scatter(
        df,
        x="latent2",
        y="latent3",
        color="sector",
        symbol="sector",
        hover_name="industry",
        hover_data=["ticker", "sector", "industry"],
        color_discrete_sequence=px.colors.qualitative.Light24,
        labels={"latent2": "PC1", "latent3": "PC2", "sector": "Sector"},
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey"), opacity=0.7),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        template="plotly_white",
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        # title='2d Visualisation of Stock Embeddings Colored By Sector'
    )
    fig.update_layout(height=550, width=650)
    fig.show()


def plot_3d(df):
    """Plot the 3D plot"""
    fig = px.scatter_3d(
        df,
        x="latent1",
        y="latent2",
        z="latent3",
        color="sector",
        hover_name="ticker",
        hover_data=["sector", "industry", "name"],
        size="size",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Light24,
    )

    fig.update_layout(
        template="plotly_white",
        title="3d Visualisation of Stock Embeddings Colored By Sector",
    )
    # fig.update_layout(height=500, width=800)
    fig.show()
