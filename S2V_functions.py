
#-- File containing a number of functions for the stock embedding project
#- For queries contact rian.dolphin@ucdconnect.ie


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------


def get_extras(returns_df, stock_df, misc_include=False):
    """
    
    Parameters
    ----------
    returns_df : Pandas DataFrame
        DESCRIPTION.
    stock_df : Pandas DataFrame
        DESCRIPTION.

    Returns
    -------
    tickers : list
        Tickers in the order of index..
    ticker2idx : dict
        Dictionary to convert from ticker to index.
    idx2ticker : dict
        Dictionary to convert from index to ticker.
    sectors : list
        Sectors in the order of index.
    industries : list
        Industries in the order of index.
    names : list
        Company names in the order of index.

    """
    #-- Get tickers list
    tickers = list(returns_df.columns)
    tickers = sorted(tickers)

    #-- Optionally remove the miscellaneous category
    #- Note: This is removed already in the returns_df_611.csv
    if not misc_include:
        for ticker in stock_df[stock_df.sector=='MISCELLANEOUS'].ticker:
            if ticker in tickers:
                returns_df.drop([ticker],axis=1, inplace=True)

    tickers = list(returns_df.columns)
    tickers = sorted(tickers)

    #-- Create dict to act as mapping between tickers and index
    #- This is useful for extracting specific embeddings from the embedding matrix
    ticker2idx = {ticker: idx for (idx, ticker) in enumerate(tickers)}
    idx2ticker = {idx: ticker for (idx, ticker) in enumerate(tickers)}


    sectors = [stock_df[stock_df.ticker==ticker].sector.values[0] for ticker in tickers]
    industries = [stock_df[stock_df.ticker==ticker].industry.values[0] for ticker in tickers]
    names = [stock_df[stock_df.ticker==ticker].name.values[0] for ticker in tickers]
    
    return tickers, ticker2idx, idx2ticker, sectors, industries, names



#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

import numpy as np
from tqdm import tqdm
#-- Get the context stocks for training

def get_idx_combinations(returns_df, tickers, CONTEXT_SIZE=3, idx_combinations=None, IQR_FILTER=False, lag=0):
    """
    This function returns a list containing target:context sets
    Inputs:
        returns_df -> A pandas dataframe of daily returns
        CONTEXT_SIZE -> The number of context stocks to be used
        idx_combinations -> Pass an existing list of idx_combinations here if you wish to add to an existing list
        IQR_FILTER -> Pass True if you wish to exclude values within the inter-quartile range (useful to get rid of noise in shorter time periods like daily)
        lag -> Introduce a lag to compare to previous returns. Default is 0.
    """
    
    #-- If not provided, initialise an empty list
    if idx_combinations==None:
        idx_combinations = []
        
    #-- Allow removal of non-movers (disregard returns in the IQR)
    #- Particularly prevalent for daily data where most stocks returns are approx 0
    #- Not neded for longer time span
    MEDIAN = np.percentile(returns_df.values.flatten(),50)
    IQR = np.percentile(returns_df.values.flatten(),75)-np.percentile(returns_df.values.flatten(),25)
    
    #-- For each row in returns_df corresponding to a date
    #- and for each stock 
    #- get the three stocks which exhibit closest returns that day
    for t in tqdm(range(lag,len(returns_df))):
        for j in range(len(tickers)):
            #-- Skip some that are 0 due to holiday day etc.
            if returns_df.iloc[t].iloc[j]==0:
                continue
            #-- Only consider cases which have returns over certain amount
            #- I.e. filter out no change days (only needed for short time horizon)
            elif IQR_FILTER & ( abs(returns_df.iloc[t].iloc[j]-MEDIAN)<IQR/2 ):
                continue
            #-- Remove extreme outliers which are entry errors/stock split etc.
            elif abs(returns_df.iloc[t].iloc[j])>10:
                continue
                
            #-- Compute the differences on a given day of query stock with each stock
            temp_differences = returns_df.iloc[t].iloc[j]-returns_df.iloc[t-lag].values
            
            #-- Set the stocks difference with itself to a very high number so it is never chosen
            temp_differences[j] = 9999

            #-- Add the indexes of the stocks with the smallest difference known as context stocks
            idx_combinations.append((list(np.argsort(abs(temp_differences))[:CONTEXT_SIZE]), j))
            #idx_combinations.append((list(np.argsort(abs(temp_differences))[3+1:3+CONTEXT_SIZE+1]), j))
            
    return idx_combinations






#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
def get_idx_combinations_hedging(returns_df, tickers, CONTEXT_SIZE=3, idx_combinations=None, IQR_FILTER=False, lag=0):
    """
    This function returns a list containing target:context sets
    Inputs:
        returns_df -> A pandas dataframe of daily returns
        CONTEXT_SIZE -> The number of context stocks to be used
        idx_combinations -> Pass an existing list of idx_combinations here if you wish to add to an existing list
        IQR_FILTER -> Pass True if you wish to exclude values within the inter-quartile range (useful to get rid of noise in shorter time periods like daily)
        lag -> Introduce a lag to compare to previous returns. Default is 0.
        
    Only two lines changed from non-hedging version
    """
    
    #-- If not provided, initialise an empty list
    if idx_combinations==None:
        idx_combinations = []
        
    #-- Allow removal of non-movers (disregard returns in the IQR)
    #- Particularly prevalent for daily data where most stocks returns are approx 0
    #- Not neded for longer time span
    MEDIAN = np.percentile(returns_df.values.flatten(),50)
    IQR = np.percentile(returns_df.values.flatten(),75)-np.percentile(returns_df.values.flatten(),25)
    
    #-- For each row in returns_df corresponding to a date
    #- and for each stock 
    #- get the three stocks which exhibit closest returns that day
    for t in tqdm(range(lag,len(returns_df))):
        for j in range(len(tickers)):
            #-- Skip some that are 0 due to holiday day etc.
            if returns_df.iloc[t].iloc[j]==0:
                continue
            #-- Only consider cases which have returns over certain amount
            #- I.e. filter out no change days (only needed for short time horizon)
            elif IQR_FILTER & ( abs(returns_df.iloc[t].iloc[j]-MEDIAN)<IQR/2 ):
                continue
            #-- Remove extreme outliers which are entry errors/stock split etc.
            elif abs(returns_df.iloc[t].iloc[j])>10:
                continue
                
            #-- Compute the differences on a given day of query stock with each stock
            # THIS LINE CHANGED FOR HEDGING
            #- Addd rather than subtract
            #- This gives ones which cancel out to give zero as best match
            #- Absolute value applied further down
            temp_differences = returns_df.iloc[t].iloc[j]+returns_df.iloc[t-lag].values

            #-- Add the indexes of the stocks with the smallest difference known as context stocks
            #- Note we must skip first stock as it will always be itself
            # THIS LINE CHANGED FOR HEDGING - dont skip itself since it won't be best anymore
            idx_combinations.append((list(np.argsort(abs(temp_differences))[0:CONTEXT_SIZE]), j))
            #idx_combinations.append((list(np.argsort(abs(temp_differences))[3+1:3+CONTEXT_SIZE+1]), j))
            
    return idx_combinations

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

def add_sets(returns_df, tickers, period, C_size = 3, idx_combinations = [], lag=0, IQR_FILTER=False, hedging=False):
    
    #-- Add returns context stocks
    temp_df = (1+returns_df).cumprod()[::period]
    temp_df = temp_df.pct_change()[1:]

    if hedging==False:
        idx_combinations = idx_combinations+get_idx_combinations(temp_df, tickers, CONTEXT_SIZE=C_size, IQR_FILTER=IQR_FILTER, lag=lag)
    #-- For hedging combinations use line below
    elif hedging:
        idx_combinations = idx_combinations+get_idx_combinations_hedging(temp_df, tickers, CONTEXT_SIZE=C_size, IQR_FILTER=IQR_FILTER, lag=lag)
    
    return idx_combinations



#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------


def get_context_sets(returns_df, tickers, CONTEXT_SIZE=3, lag=0, save=True, periods=['daily','weekly', 'monthly'],
                     IQR_daily=True, hedging=False):
    #-- Get the target:context sets
    idx_combinations = []
    

    print("Getting TGT:CONTEXT Sets")
    if 'daily' in periods:
        idx_combinations = add_sets(returns_df, tickers, period=1, C_size=CONTEXT_SIZE, idx_combinations=idx_combinations, lag=lag,
                                   IQR_FILTER=IQR_daily, hedging=hedging)
        print("Daily pairs added")
        
    if 'weekly' in periods:
        idx_combinations = add_sets(returns_df, tickers, period=5, C_size=CONTEXT_SIZE, idx_combinations=idx_combinations, lag=lag, hedging=hedging)
        print("Weekly pairs added")
        
    if 'monthly' in periods:
        idx_combinations = add_sets(returns_df, tickers, period=21, C_size=CONTEXT_SIZE, idx_combinations=idx_combinations, lag=lag, hedging=hedging)
        print("Monthly pairs added")
        
    
    #-- Generate a save string
    s = ''
    for xi in periods:
        s+="_"+xi
    
    if not hedging:
        save_string = f'TrainingData/IJCNN_TP{s}_L{lag}_C{CONTEXT_SIZE}_IQRDaily{str(IQR_daily)}_Training_0to70.json'
    elif hedging:
        save_string = f'TrainingData/IJCNN_TP{s}_L{lag}_C{CONTEXT_SIZE}_IQRDaily{str(IQR_daily)}_Training_0to70_HEDGING.json'
    
    if save:
        df=pd.DataFrame([[idx_combinations[i][0] for i in range(len(idx_combinations))], [idx_combinations[i][1] for i in range(len(idx_combinations))]])
        df=df.T
        df.head() 
        df.to_json(save_string)
        
    print("--- DONE ---")
    print(f'Number of context sets: {len(idx_combinations)}')
    if save:
        print(f'Saved to: {save_string}')
        
    return idx_combinations

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
import pandas as pd
def get_cooccurrence_df(tickers, idx_combinations, idx2ticker):
    cooccurrence_dict = {}
    nan_array = np.array([0]*len(tickers))
    for ticker in tickers:
        cooccurrence_dict[ticker]=np.copy(nan_array)


    #-- Count the cooccurences for each ticker
    temp = []
    for i in tqdm(range(len(idx_combinations))):
        temp_idx=idx_combinations[i][1]
        temp_ticker = idx2ticker[temp_idx]
        temp_context_idxs = idx_combinations[i][0]
        for idx in temp_context_idxs:
            if idx==temp_idx:
                print(idx_combinations[i])
            cooccurrence_dict[temp_ticker][idx]+=1

    cooccurrence_df = pd.DataFrame(cooccurrence_dict, index=tickers)
    
    return cooccurrence_df

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

import torch
    
def get_train_loader(idx_combinations):
    
    x_vals = np.array([idx_combinations[i][0] for i in range(len(idx_combinations))])
    y_vals = np.array([idx_combinations[i][1] for i in range(len(idx_combinations))])

    x_train_tensor = torch.from_numpy(x_vals)#.float()
    y_train_tensor = torch.from_numpy(y_vals)#.float()

    train_data = CustomDataset(x_train_tensor, y_train_tensor)


    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, drop_last=True)
    
    return train_loader

#train_loader = get_train_loader(idx_combinations=idx_combinations)



#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
import torch.optim as optim
import torch.nn as nn
from S2V_models import CBOW_StockModeller_Single, CBOW_StockModeller_Double

def get_model(EMBEDDING_DIM, tickers, model_type=0):

    #torch.manual_seed(42)
    
    if model_type==0:
        model = CBOW_StockModeller_Single(len(tickers), EMBEDDING_DIM)
    elif model_type==1:
        model = CBOW_StockModeller_Double(len(tickers), EMBEDDING_DIM)
    
    #-- Define loss function
    #- Note: NLLLoss() used on a log_softmax output is the same as cross entropy loss
    loss_function = nn.NLLLoss() 
    
    
    optimizer = optim.Adam(model.parameters(),
                           lr=0.0001, 
                           #weight_decay=0.01
                          )
    return model, loss_function, optimizer

#model, loss_function, optimizer = get_model(EMBEDDING_DIM=10, CONTEXT_SIZE=3)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

def train_model(epochs, model, loss_function, optimizer, train_loader, early_stop=True):
    """
    Parameters
    ----------
    epochs : int
    model : PyTorch Model
    loss_function : 
    optimizer : 
    train_loader : 
    early_stop : bool, optional
        Whether to stop training when loss no longer being reduced. The default is True.

    Returns
    -------
    losses : TYPE
        DESCRIPTION.

    """
    losses=[]
    #-- Training Loop
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        count=0
        for x_batch, y_batch in train_loader:
            count+=1

            #-- Zero out gradients (PyTorch accumulates gradients otherwise)
            model.zero_grad()

            #-- Run the forward pass, getting log probabilities output for the current batch
            log_probs = model(x_batch)

            #-- Compute the loss function
            loss = loss_function(log_probs.squeeze(), y_batch)

            #-- Backward Pass
            loss.backward()
            #-- Update Gradient
            optimizer.step()

            #-- Add current loss for the batch to the total epoch loss
            total_loss += loss.item()

        #-- Add epoch loss to the list for plotting
        losses.append(total_loss/count)
        
        if (epoch>2) & (early_stop):
            if abs(losses[-1]-losses[-2])<0.0002:
                print(f'Training stopped at epoch {epoch} as loss reduction threshold reached')
                return losses

    return losses
#losses = train_model(10)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

def train_model_weights(epochs, model, weights_df, idx2ticker, loss_function, optimizer, train_loader, early_stop=True):
    """
    Parameters
    ----------
    epochs : int
    model : PyTorch Model
    loss_function : 
    optimizer : 
    train_loader : 
    early_stop : bool, optional
        Whether to stop training when loss no longer being reduced. The default is True.

    Returns
    -------
    losses : TYPE
        DESCRIPTION.

    """
    losses=[]
    #-- Training Loop
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        count=0
        for x_batch, y_batch in train_loader:
            count+=1

            #-- Zero out gradients (PyTorch accumulates gradients otherwise)
            model.zero_grad()

            #-- Run the forward pass, getting log probabilities output for the current batch
            log_probs = model(x_batch, y_batch, weights_df, idx2ticker)

            #-- Compute the loss function
            loss = loss_function(log_probs.squeeze(), y_batch)

            #-- Backward Pass
            loss.backward()
            #-- Update Gradient
            optimizer.step()

            #-- Add current loss for the batch to the total epoch loss
            total_loss += loss.item()

        #-- Add epoch loss to the list for plotting
        losses.append(total_loss/count)
        
        if (epoch>2) & (early_stop):
            if abs(losses[-1]-losses[-2])<0.0002:
                print(f'Training stopped at epoch {epoch} as loss reduction threshold reached')
                return losses

    return losses
#losses = train_model(10)


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

import plotly.graph_objects as go
def plot_losses(losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(losses))+1, y=np.array(losses)[:]))
    fig.update_layout(template='plotly_white',
                      title='Training Loss',
                      xaxis=dict(title='Epoch'),
                      yaxis=dict(title='Loss'),
                      width=650,
                      height=400)
    fig.show()

#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

#-- Sector Prediction Evaluation
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def eval_sector_predict(model, tickers, stock_df, embedding_matrix):
    #embedding_matrix = model.embeddings.weight.detach().numpy()

    sectors = [stock_df[stock_df.ticker==ticker].sector.values[0] for ticker in tickers]

    X_train, X_test, y_train, y_test = train_test_split(embedding_matrix, sectors, test_size=0.3, random_state=42)

    #classifier = MLPClassifier(hidden_layer_sizes=[120, 64, 32], batch_size=20, max_iter=1000)
    classifier = SVC()
    classifier.fit(X_train,y_train)
    
    y_pred = classifier.predict(X_test)

    #return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)
    return None, accuracy_score(y_test, y_pred)


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------

def overall(idx_combinations, tickers, stock_df, CONTEXT_SIZE, EMBEDDING_DIM, EPOCHS, PERIOD, plot_loss=True, EARLY_STOP=True, model_type=0):
    """
    CONTEXT_SIZE - Only needed if creating the idx_combinations in this function. Faster to create them first and then reuse
    EMBEDDING_DIM - Needed for the get_model function to define the embedding matrix.
    EPOCHS - number of epochs for training
    PERIOD - Only needed if creating the idx_combinations in this function. Faster to create them first and then reuse
    plot_loss - Boolean. Whether to produce a plot of training loss by epoch or not. Default True.
    """
    print("=="*20)
    print(f'C={CONTEXT_SIZE}, N={EMBEDDING_DIM}, Epochs={EPOCHS}, Period={PERIOD}')
    print("=="*20)

#     #-- Get the target:context sets
#     idx_combinations = []

#     print("Getting TGT:CONTEXT Sets")
#     idx_combinations = add_sets(period=PERIOD, C_size=CONTEXT_SIZE, idx_combinations=idx_combinations)
#     idx_combinations = add_sets(period=5, C_size=CONTEXT_SIZE, idx_combinations=idx_combinations)
#     idx_combinations = add_sets(period=21, C_size=CONTEXT_SIZE, idx_combinations=idx_combinations)
#     print(f'Completed: {len(idx_combinations)} TGT:CONTEXT Sets')

    #-- Create the PyTorch Train Loader
    train_loader = get_train_loader(idx_combinations=idx_combinations)

    #-- Define Model
    model, loss_function, optimizer = get_model(EMBEDDING_DIM=EMBEDDING_DIM, tickers=tickers, model_type=model_type)

    #-- Train the model
    print("Training Model")
    losses = train_model(EPOCHS, model, loss_function, optimizer, train_loader, early_stop=EARLY_STOP)
    print("Training Completed")

    #-- Plot Losses
    if plot_loss:
        plot_losses(losses)
    
    #-- Store the output of the evaluation function
    eval_out = eval_sector_predict(model, tickers, stock_df, embedding_matrix=model.embeddings.weight.detach().numpy())
    
    #-- Evaluate sector prediction task
    #print(eval_out[0])
    print(f'Accuracy: {eval_out[1]}')
    
    return model
