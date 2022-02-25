#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:50:56 2021

@author: rian
"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
def pca_plot_from_embeddings(embedding_matrix, sectors, tickers,industries, names,
                             dimensions=2, reduced=True, method='PCA', return_df=False, rand_state=None):
    """

    Parameters
    ----------
    model : Pytorch Stock Modeller
        Model instance of the previously defined class
    sectors : List of sectors
    tickers : List of tickers
    industries : List of industries
    names : list of names
    dimensions : int, optional
        Number of dimensions for PCA plot, either 2 or 3. The default is 2.
    reduced : Bool, optional
        Whether to plot all sectors or just a reduced set to avoid cluttered plot. The default is True.
        
    return_df : Bool
    
    rand_state : int, Default None
        Only used if TSNE

    Returns
    -------
    None. Produces a plot.

    """
    
    #embedding_matrix = model.embeddings.weight.detach().numpy() #-- This is now an input

    
    
    
    X = embedding_matrix
    
    
    if method=='PCA':
        vals = PCA(n_components=3).fit_transform(X)
    elif method=='TSNE':
        vals = TSNE(n_components=3, random_state=rand_state).fit_transform(X)
    #pca.fit(X)

    #print(pca.explained_variance_ratio_)

    #print(pca.singular_values_)

    #vals = pca.transform(X)
    
    
    
    plot_df = pd.DataFrame(data={'latent1':vals[:,0],
              'latent2':vals[:,1],
              'latent3':vals[:,2],
              #'PCA4':vals[:,3],
              "sector":sectors,
              "ticker":tickers,
              "industry":industries,
              "name":names,
              "size":[0.5 for _ in range(len(names))]})
    
    
    if reduced:
            sub_df = plot_df[plot_df.sector.isin(['TECHNOLOGY','FINANCE','ENERGY','PUBLIC UTILITIES'])]
    else:
        sub_df = plot_df

    if dimensions==2:
        
        fig = px.scatter(sub_df,
                     x="latent2", y="latent3", color="sector", symbol="sector",
              hover_name="industry", hover_data=["ticker","sector", "industry"],
                    color_discrete_sequence=px.colors.qualitative.Light24,
                    labels={"latent2":"PC1",
                            "latent3":"PC2",
                            "sector":"Sector"})
        fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))
        fig.update_traces(marker=dict(size=8,
                              line=dict(width=1,
                                        color='DarkSlateGrey'),
                              opacity=0.7),
                  selector=dict(mode='markers'))


        fig.update_layout(template='plotly_white',
                         #paper_bgcolor='rgba(0,0,0,0)',
                         #plot_bgcolor='rgba(0,0,0,0)',
                          #title='2d Visualisation of Stock Embeddings Colored By Sector'
                         )
        fig.update_layout(height=550, width=650)
        fig.show()
    elif dimensions==3:
        fig = px.scatter_3d(sub_df, x='latent1', y='latent2', z='latent3',
                      color='sector', hover_name="ticker", hover_data=["sector", "industry", "name"],
                           size="size", opacity=0.7,
                           color_discrete_sequence=px.colors.qualitative.Light24)
        
        fig.update_layout(template='plotly_white',
                         title='3d Visualisation of Stock Embeddings Colored By Sector')
        #fig.update_layout(height=500, width=800)
        fig.show()
    
    if return_df:
        return plot_df