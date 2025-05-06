import numpy as np
import pandas as pd
from scipy.io import loadmat
from typing import List, Tuple
import plotly.graph_objects as go

# Load the time series data from provided .mat file
def load_time_series(mat_file_path: str, variable_name: str = 'Xtrain') -> np.ndarray:
    mat = loadmat(mat_file_path) # Load the file as a dictionary
    return mat[variable_name].flatten()  # Flatten to 1D array for easy indexing

# Given a 1D array create a dataframe that contains the target value (y) and columns for each lag
def create_lagged_features(series: np.ndarray, n_lags: int, dropna: bool = True) -> pd.DataFrame:
    df = pd.DataFrame({'y': series})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    
    if dropna:
        df = df.dropna().reset_index(drop=True)  # <-- fixed: assigned the cleaned DataFrame back
    
    return df


# Given the 'lagged' dataframe, split them into training and testing, if the size is small enough
# for multiple 'windows' to be taken from original data make those as well. The split percentage
# is configurable.

# The 'total_size' is the number of consecutive rows (lagged time steps) to include in a set of 
# data (train + est). The size of the 'chunk' you'll train test on. The 'stride' is how far
# forward in the data to move to start the next 'chunk'. If stride < total_size you get 
# (partially) overlapping chunks, if stride = total_size you get non overlapping chunks and if
# stride > total_size you skip some data between chunks.

# It returns a list of tuples for these 'chunks'
def train_test_split_lagged(
    df: pd.DataFrame, # DataFrame with lagged features.
    train_size: float, # Percentage used for training
    total_size: int = None, # 
    stride: int = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    if total_size is not None:
        max_start = len(df) - total_size
        if stride is None:
            stride = total_size  # No overlap
        splits = []
        for start in range(0, max_start + 1, stride):
            chunk = df.iloc[start:start + total_size]
            split_idx = int(len(chunk) * train_size)
            train_df = chunk.iloc[:split_idx]
            test_df = chunk.iloc[split_idx:]
            splits.append((train_df, test_df))
        return splits
    else:
        split_idx = int(len(df) * train_size)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        return [(train_df, test_df)]

# Plot the full time series and highlight the chunks as provided by the function:
# 'train_test_split_lagged'.

# The n_lags and stride must correspond with the settings used for split otherwise weird
# behaviour.
def plot_lagged_chunks(series: np.ndarray, splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
                       n_lags: int, stride: int):
    
    # Below is adapted from Niek's code to visualise the series
    X_train_list = series.tolist()
    x_full = list(range(len(X_train_list)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_full,
        y=X_train_list,
        mode='lines',
        name='Full Series',
        line=dict(color='lightgray')
    ))

    # Go over all the splits and determine between which indices the entries fall in the chunk
    # then plot them.
    for i, (train_df, test_df) in enumerate(splits):
        # Account for lag-induced trimming at the start
        start_idx = i * stride + n_lags
        chunk_size = len(train_df) + len(test_df)

        # Train chunk
        x_train = list(range(start_idx, start_idx + len(train_df)))
        y_train = X_train_list[start_idx : start_idx + len(train_df)]

        # Test chunk
        x_test = list(range(start_idx + len(train_df), start_idx + chunk_size))
        y_test = X_train_list[start_idx + len(train_df) : start_idx + chunk_size]

        fig.add_trace(go.Scatter(
            x=x_train,
            y=y_train,
            mode='lines',
            name=f'Train Chunk {i+1}',
            line=dict(color='green', width=2),
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=x_test,
            y=y_test,
            mode='lines',
            name=f'Test Chunk {i+1}',
            line=dict(color='orange', width=2, dash='dot'),
            opacity=0.7
        ))

    fig.update_layout(
        title='X_train Time Series with Highlighted Chunks',
        xaxis_title='Index',
        yaxis_title='Value',
        legend_title='Data Segments',
        height=500
    )
    fig.show()