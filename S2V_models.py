#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:32:38 2021

@author: rian
"""

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch
class CBOW_StockModeller_Single(nn.Module):
    """
    Model architecture similar to CBOW Word2Vec but adapted for stock modelling
    Note: Context size is inferred from the inputs and is not needed to be defined here
    """
    def __init__(self, n_tickers, embedding_dim):
        super(CBOW_StockModeller_Single, self).__init__() #-- This line calls the parent class nn.Module
        #-- Only use one embedding matrix here
        #- Seperate Input and Output embeddings not needed in this application
        #- In NLP Word2Vec there is input and output matrix
        self.embeddings = nn.Embedding(n_tickers, embedding_dim)
        
    def forward(self, inputs):
        #-- This extracts the relevant rows of the embedding matrix
        #- Equivalent to W^T x_i in "word2vec Parameter Learning Explained"
        temp = self.embeddings(inputs)#.view((len(inputs),-1))
        
        #-- Compute the hidden layer by a simple mean
        hidden = temp.mean(axis=1)
        #-- Reshape to make matrix dimensions compatible
        hidden = hidden.unsqueeze(dim=2)
        #-- Compute dot product of hidden with embeddings
        out = torch.matmul(self.embeddings.weight, hidden)
        
        #-- Return the log softmax since we use NLLLoss loss function
        return F.log_softmax(out, dim=1)
    
    
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
    
import numpy as np
class CBOW_StockModeller_Single_Weights(nn.Module):
    """
    Model architecture similar to CBOW Word2Vec but adapted for stock modelling
    Note: Context size is inferred from the inputs and is not needed to be defined here
    """
    def __init__(self, n_tickers, embedding_dim):
        super(CBOW_StockModeller_Single_Weights, self).__init__() #-- This line calls the parent class nn.Module
        #-- Only use one embedding matrix here
        #- Seperate Input and Output embeddings not needed in this application
        #- In NLP Word2Vec there is input and output matrix
        self.embeddings = nn.Embedding(n_tickers, embedding_dim)
        
    def forward(self, inputs, y_batch, weights_df, idx2ticker):
        """

        Parameters
        ----------
        inputs : train_loader
            The batch for training.
        weights_df : DataFrame
            The dataframe containing the count of how many times each stock appears
            in anothers context. From this the weights will be computed.
        idx2ticker : Dictionary
            Dictionary to map from index value to ticker since the weights_df has tickers
            as the column names and axes.

        Returns
        -------
        Log Softmax prediction for the target stocks

        """
        #-- This extracts the relevant rows of the embedding matrix
        #- Equivalent to W^T x_i in "word2vec Parameter Learning Explained"
        temp = self.embeddings(inputs)#.view((len(inputs),-1))
        
        
        
        """
        This won't work unless the y_batch is given while training - is this reasonable?
        Yes maybe, since we are not going to be using for predictions.
        """
        
        #-- Get the weights for the input batch
        w_list = []
        for i in range(len(inputs)):
            w = weights_df[idx2ticker[y_batch[i].item()]].iloc[inputs[i].tolist()].values
            w = w/sum(w)
            w_list.append(list(w))
            
        #-- Store the weights for the whole batch
        input_weights = torch.from_numpy(np.array(w_list))
        
        #-- Multiply the weights in
        temp = temp*input_weights[:,:,None]
        
        #-- Compute the hidden layer
        #- Since the weights are already multiplied in we just sum
        hidden = temp.sum(axis=1)
        
        cosine=True
        if not cosine:
            #-- Reshape to make matrix dimensions compatible
            hidden = hidden.unsqueeze(dim=2)
            #-- Compute dot product of hidden with embeddings
            out = torch.matmul(self.embeddings.weight.double(), hidden.double())
        else:
            #-- Compute cosine similarity
            out = sim_matrix(self.embeddings.weight, hidden) #-- sim_matrix defined below
            #-- Swap rows and columns as required
            #out = out.reshape(temp.shape[0],-1)
            out = torch.t(out)
            #-- Unsqueeze to get in correct format
            out = out.unsqueeze(dim=2)
        
        #-- Return the log softmax since we use NLLLoss loss function
        return F.log_softmax(out, dim=1)
    
    
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm.double(), b_norm.transpose(0, 1).double())
    return sim_mt
    
    
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------


class CBOW_StockModeller_Single_Weights_Added_Layers(nn.Module):
    """
    Model architecture similar to CBOW Word2Vec but adapted for stock modelling
    Note: Context size is inferred from the inputs and is not needed to be defined here
    """
    def __init__(self, n_tickers, embedding_dim):
        super(CBOW_StockModeller_Single_Weights, self).__init__() #-- This line calls the parent class nn.Module
        #-- Only use one embedding matrix here
        #- Seperate Input and Output embeddings not needed in this application
        #- In NLP Word2Vec there is input and output matrix
        self.embeddings = nn.Embedding(n_tickers, embedding_dim)
        
        #-- FOR NEW LAYERS
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, inputs, y_batch, weights_df, idx2ticker):
        """

        Parameters
        ----------
        inputs : train_loader
            The batch for training.
        weights_df : DataFrame
            The dataframe containing the count of how many times each stock appears
            in anothers context. From this the weights will be computed.
        idx2ticker : Dictionary
            Dictionary to map from index value to ticker since the weights_df has tickers
            as the column names and axes.

        Returns
        -------
        Log Softmax prediction for the target stocks

        """
        #-- This extracts the relevant rows of the embedding matrix
        #- Equivalent to W^T x_i in "word2vec Parameter Learning Explained"
        temp = self.embeddings(inputs)#.view((len(inputs),-1))
        """
        This won't work unless the y_batch is given while training - is this reasonable?
        Yes maybe, since we are not going to be using for predictions.
        """
        
        #-- Get the weights for the input batch
        w_list = []
        for i in range(len(inputs)):
            w = weights_df[idx2ticker[y_batch[i].item()]].iloc[inputs[i].tolist()].values
            w = w/sum(w)
            w_list.append(list(w))
            
        #-- Store the weights for the whole batch
        input_weights = torch.from_numpy(np.array(w_list))
        
        #-- Multiply the weights in
        temp = temp*input_weights[:,:,None]
        
        #-- Compute the hidden layer
        #- We can use a simple mean since the weights are already multiplied in
        hidden = temp.mean(axis=1)
        #-- Reshape to make matrix dimensions compatible
        hidden = hidden.unsqueeze(dim=2)
        #-- Compute dot product of hidden with embeddings
        out = torch.matmul(self.embeddings.weight.double(), hidden.double())
        
        #-- Return the log softmax since we use NLLLoss loss function
        return F.log_softmax(out, dim=1)
    
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

class CBOW_StockModeller_Double(nn.Module):
    """
    Model architecture similar to CBOW Word2Vec but adapted for stock modelling
    Note: Context size is inferred from the inputs and is not needed to be defined here
    """
    def __init__(self, n_tickers, embedding_dim):
        super(CBOW_StockModeller_Double, self).__init__() #-- This line calls the parent class nn.Module
        #-- Only use one embedding matrix here
        #- Seperate Input and Output embeddings not needed in this application
        #- In NLP Word2Vec there is input and output matrix
        self.embeddings_in = nn.Embedding(n_tickers, embedding_dim)
        self.embeddings_out = nn.Embedding(n_tickers, embedding_dim)
        
    def forward(self, inputs):
        #-- This gets the relevant rows of the embedding matrix
        #- Equivalent to (1/C) W^T (x_1+x_2+...+x_C) in "word2vec Parameter Learning Explained"
        hidden = self.embeddings_in(inputs).mean(axis=1)
        
        #-- Reshape to make matrix dimensions compatible
        hidden = hidden.unsqueeze(dim=2)
        #-- Compute dot product of hidden with embeddings
        #- W_out * h
        out = torch.matmul(self.embeddings_out.weight, hidden)
        
        #-- Return the log softmax since we use NLLLoss loss function
        return F.log_softmax(out, dim=1)
    
    
    
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
    
#-- Skip gram