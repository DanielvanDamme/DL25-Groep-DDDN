import optuna
import torch
import numpy as np
from pytorch_tcn import TCN
import statistics
import json
from timeseries_split import (
    load_time_series,
    create_lagged_features,
    train_test_split_lagged,
    plot_lagged_chunks
)

def get_data(n_lags, series):
    """
    1. Splits the training data up in 5 folds
    2. Splits those folds up training and test data using a 80/20 split
    3. Tensorizes data so that it can be inputted in the model

    Parameters:
    n_lags (int): number of preceding steps used for prediction
    series (np.array): normalized series representing training data

    Returns:
    List[List[torch.FloatTensor]]): A list of folds, each fold is a list of [X_train, y_train, X_test, y_test]
    """
    lagged_df = create_lagged_features(series=series, n_lags=n_lags, dropna=True)


    splits = train_test_split_lagged(df=lagged_df, train_size=0.8, total_size=200, stride=200)

    out_data = []
    accumulated_X_train = None
    accumulated_y_train = None

    prev_test_set = []  # To store all previous test sets

    for split in splits:
        # Tensorize training data
        train_data = split[0].dropna()
        X_train_df = train_data.drop("y", axis=1)
        y_train_df = train_data["y"]
        X_train = torch.FloatTensor(X_train_df.values).transpose(0, 1)
        y_train = torch.FloatTensor(y_train_df.values)

        # Tensorize test data
        test_data = split[1].dropna()
        X_test_df = test_data.drop("y", axis=1)
        y_test_df = test_data["y"]
        X_test = torch.FloatTensor(X_test_df.values).transpose(0, 1)
        y_test = torch.FloatTensor(y_test_df.values)

        # Add previous test set to current train data
        if prev_test_set:
            for prev_X_test, prev_y_test in prev_test_set:
                X_train = torch.cat((X_train, prev_X_test), dim=1)
                y_train = torch.cat((y_train, prev_y_test), dim=0)

        # Add current training data to the total training data
        if accumulated_X_train is None:
            accumulated_X_train = X_train
            accumulated_y_train = y_train
        else:
            accumulated_X_train = torch.cat((accumulated_X_train, X_train), dim=1)
            accumulated_y_train = torch.cat((accumulated_y_train, y_train), dim=0)

        # Store the current test set for use in next folds
        prev_test_set = [(X_test, y_test)]

        # Append to output
        out_data.append([accumulated_X_train.clone(), accumulated_y_train.clone(), X_test, y_test])

    return out_data

def setup_model(n_lags, n_channels, n_layers, kernel_size):
    """
    Instantiates the model with the parameters suggested by the current trial.

    Parameters:
    n_lags (int): number of preceding steps used for prediction
    n_channels (int): number of channels per layer in model
    n_layers (int): number of layers in model
    kernel_size: size of convolutional filter

    Returns:
    <class 'tcn.TCN'>: An instantiated pytorch model

    """
    model = TCN(
    num_inputs=n_lags,
    num_channels=[n_channels] * n_layers, # num_channels wants number of channels per layer, we use the same amount of channels for each layer
    kernel_size=kernel_size,
    output_projection=1 # We want one number as the output
    )
    return model

def train_test_loop(folds, model, optimizer, criterion, fold_weights):
    """
    This function trains the model and evaluates it on the test set. It returns
    the mean of the final test losses per fold, where each test loss is weighted
    by the normalized size of the training set.

    Parameters:
    folds (List[List[torch.FloatTensor]]): A list of folds, each fold is a list of [X_train, y_train, X_test, y_test]
    model (<class 'tcn.TCN'>): An instantiated pytorch model
    optimizer (torch.optim.adam.Adam): the Adam optimizer instantiated with a chosen learning rate
    criterion (torch.nn.MSELoss): The MSE loss function
    fold_weights (numpy.ndarray): list of weights per fold

    Returns: 
    float: the mean of the final test losses per fold, where each test loss is weighted
    by the normalized size of the training set.
    """
    epochs = 10 # each model is trained for only 10 epochs to speed up the search
    test_losses = []

    for i,fold in enumerate(folds):
        X_train, y_train, X_test, y_test = fold

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad() # zero gradients
            preds = model(X_train)
            loss = criterion(preds.squeeze(), y_train) # calculate training loss
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = criterion(test_preds.squeeze(), y_test)
            weighted_loss = fold_weights[i] * test_loss.item()  # weight the loss by normalized training set size
            test_losses.append(weighted_loss)
    
    return statistics.mean(test_losses) # return the average of the weighted test losses


def objective(trial):
    """
    This function runs the hyperparameter tuning. A range of possible hyperparameters is suggested,
    and a score is calculated for how good they are.

    Parameters:
    trial int: what trial is being run

    Returns:
    float: the mean of the final test losses per fold, where each test loss is weighted
    by the normalized size of the training set.
    """
    # Hyperparameters to be tuned, for each a range is defined where to search in
    n_lags = trial.suggest_int("n_lags", 10, 60, step=10)
    n_channels = trial.suggest_int("n_channels", 2, 40, step=2)
    n_layers = trial.suggest_int("n_layers", 2, 5)
    kernel_size = trial.suggest_int("kernel_size", 3, 9, step=2)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    
    folds = get_data(n_lags, series)
    #determine fold weights: get a list of the training fold sizes and normalize it (so they add up to 1)
    sizes = [len(fold[0]) for fold in folds]
    weights = [x / sum(sizes) for x in sizes]
    model = setup_model(n_lags, n_channels, n_layers, kernel_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    score = train_test_loop(folds,model,optimizer,criterion, weights)
    return score


# Load and prepare
series = load_time_series("assignment 1\code\Xtrain.mat")
# Normalization
series = (series - np.mean(series)) / np.std(series)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=2000) # run the search for 2000 iterations

print("Best parameters", study.best_params)


with open("assignment 1/code/tcnn_model/tcnn_info/best_params.json", "w") as f:
    json.dump(study.best_params, f)