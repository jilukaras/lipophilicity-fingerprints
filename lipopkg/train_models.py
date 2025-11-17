import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from .fingerprints import smiles_to_mols, mols_to_morgan, mols_to_maccs

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    print("Working directory: " + os.getcwd())
    print("Python executable: " + sys.executable)

    env = os.getenv("CONDA_DEFAULT_ENV")
    if env is None:
        env = "not set"
    print("Conda environment: " + env)
    print("")

    # CSV is inside the same package folder
    csv_path = os.path.join(os.path.dirname(__file__), "Lipophilicity.csv")

    if not os.path.exists(csv_path):
        print("Could not find " + csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    print("Columns in CSV:")
    print(list(df.columns))
    print("")

    smiles_col = "smiles"
    target_col = "exp"

    if smiles_col not in df.columns or target_col not in df.columns:
        print("Expected columns 'smiles' and 'exp' in CSV.")
        sys.exit(1)

    df = df[[smiles_col, target_col]].dropna().reset_index(drop=True)

    smiles_list = df[smiles_col].tolist()
    y = df[target_col].values.astype(float)

    mols = smiles_to_mols(smiles_list)

    mols_train, mols_test, y_train, y_test = train_test_split(
        mols, y, test_size=0.2, random_state=42
    )

    print("Train size: " + str(len(mols_train)))
    print("Test size: " + str(len(mols_test)))
    print("")

    print("Building Morgan fingerprints...")
    X_train_morgan = mols_to_morgan(mols_train, n_bits=2048, radius=2)
    X_test_morgan = mols_to_morgan(mols_test, n_bits=2048, radius=2)

    print("Building MACCS keys...")
    X_train_maccs = mols_to_maccs(mols_train)
    X_test_maccs = mols_to_maccs(mols_test)

    print("Morgan train shape: " + str(X_train_morgan.shape))
    print("Morgan test shape: " + str(X_test_morgan.shape))
    print("MACCS train shape: " + str(X_train_maccs.shape))
    print("MACCS test shape: " + str(X_test_maccs.shape))
    print("")

    print("Scaling targets...")
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    mlp_morgan = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        max_iter=500,
        early_stopping=True,
        random_state=42
    )

    mlp_maccs = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        max_iter=500,
        early_stopping=True,
        random_state=42
    )

    print("Training MLP on Morgan fingerprints...")
    mlp_morgan.fit(X_train_morgan, y_train_scaled)

    print("Training MLP on MACCS keys...")
    mlp_maccs.fit(X_train_maccs, y_train_scaled)

    print("")
    print("Evaluating models...")

    y_pred_morgan_scaled = mlp_morgan.predict(X_test_morgan)
    y_pred_morgan = y_scaler.inverse_transform(
        y_pred_morgan_scaled.reshape(-1, 1)
    ).ravel()

    y_pred_maccs_scaled = mlp_maccs.predict(X_test_maccs)
    y_pred_maccs = y_scaler.inverse_transform(
        y_pred_maccs_scaled.reshape(-1, 1)
    ).ravel()

    rmse_morgan = rmse(y_test, y_pred_morgan)
    rmse_maccs = rmse(y_test, y_pred_maccs)

    print("")
    print("=== RMSE on unscaled targets ===")
    print("Morgan fingerprints RMSE: " + str(rmse_morgan))
    print("MACCS keys RMSE: " + str(rmse_maccs))
    print("Conda environment (again): " + env)

    if rmse_morgan < rmse_maccs:
        print("Morgan fingerprints performed better on this split.")
    elif rmse_morgan > rmse_maccs:
        print("MACCS keys performed better on this split.")
    else:
        print("Both feature sets have the same RMSE on this split.")

if __name__ == "__main__":
    main()
