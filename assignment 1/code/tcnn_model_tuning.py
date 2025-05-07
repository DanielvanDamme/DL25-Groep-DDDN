import optuna
import torch
import numpy as np
from pytorch_tcn import TCN
import statistics
from timeseries_split import (
    load_time_series,
    create_lagged_features,
    train_test_split_lagged,
    plot_lagged_chunks
)

# Load and prepare
series = load_time_series("assignment 1\code\Xtrain.mat")
# Normalization
series = (series - np.mean(series)) / np.std(series)

def get_data(n_lags):
    """
    For each fold, concatenate current and all preceding training data,
    including testing data from all preceding folds.
    Testing data for the current fold remains unchanged.
    """
    lagged_df = create_lagged_features(series=series, n_lags=n_lags, dropna=True)


    splits = train_test_split_lagged(df=lagged_df, train_size=0.8, total_size=200, stride=200)

    out_data = []
    accumulated_X_train = None
    accumulated_y_train = None

    prev_test_set = []  # To store all previous test sets

    for split in splits:
        # Clean and extract current training data
        train_data = split[0].dropna()
        X_train_df = train_data.drop("y", axis=1)
        y_train_df = train_data["y"]
        X_train = torch.FloatTensor(X_train_df.values).transpose(0, 1)
        y_train = torch.FloatTensor(y_train_df.values)

        # Clean and extract current test data
        test_data = split[1].dropna()
        X_test_df = test_data.drop("y", axis=1)
        y_test_df = test_data["y"]
        X_test = torch.FloatTensor(X_test_df.values).transpose(0, 1)
        y_test = torch.FloatTensor(y_test_df.values)

        # Include previous test data into accumulated training data
        if prev_test_set:
            for prev_X_test, prev_y_test in prev_test_set:
                X_train = torch.cat((X_train, prev_X_test), dim=1)
                y_train = torch.cat((y_train, prev_y_test), dim=0)

        # Accumulate training data
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

    model = TCN(
    num_inputs=n_lags,
    num_channels=[n_channels] * n_layers, 
    kernel_size=kernel_size,
    output_projection=1
    )
    return model

def train_test_loop(folds, model, optimizer, criterion):
    epochs = 10
    test_losses = []

    for fold in folds:
        X_train, y_train, X_test, y_test = fold

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train)
            loss = criterion(preds.squeeze(), y_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = criterion(test_preds.squeeze(), y_test)
            test_losses.append(test_loss.item())
    
    return statistics.mean(test_losses)


def objective(trial):

    # Hyperparameters to be tuned
    n_lags = trial.suggest_int("n_lags", 10, 60, step=10)
    n_channels = trial.suggest_int("n_channels", 3, 10)
    n_layers = trial.suggest_int("n_layers", 2, 5)
    kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    
    folds = get_data(n_lags)
    model = setup_model(n_lags, n_channels, n_layers, kernel_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    score = train_test_loop(folds,model,optimizer,criterion)
    return score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best parameters", study.best_params)