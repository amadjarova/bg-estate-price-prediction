import os
import pickle
import pandas as pd

from src.models.cart import CARTRegressor
from src.utils.utils import calculate_metrics, train_test_split
from src.models.random_forest import RandomForestRegressor
from src.models.knn import KNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models_saved')
RF_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
KNN_PATH = os.path.join(MODEL_DIR, 'knn_model.pkl')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def run_detailed_validation(rf, knn, processed_csv):
    train_df = pd.read_csv(processed_csv).dropna()
    X = train_df.drop('Price', axis=1).values.tolist()
    y = train_df['Price'].values.tolist()

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)

    p_rf = rf.predict(X_test)
    p_knn = knn.predict(X_test)

    cart = CARTRegressor(max_depth=10)

    X_train_temp, _, y_train_temp, _ = train_test_split(X, y, test_size=0.2)
    cart.fit(X_train_temp, y_train_temp)
    p_cart = cart.predict(X_test)

    mae_rf, mape_rf = calculate_metrics(y_test, p_rf)
    mae_knn, mape_knn = calculate_metrics(y_test, p_knn)
    mae_cart, mape_cart = calculate_metrics(y_test, p_cart)

    p_final = [0.7 * p_rf[i] + 0.3 * p_knn[i] for i in range(len(y_test))]
    mae_final, mape_final = calculate_metrics(y_test, p_final)

    print("\n" + "=" * 55)
    print(f"{'ALGORITHM':<25} | {'MAE (€)':<12} | {'ACCURACY (%)'}")
    print("-" * 55)
    print(f"{'Random Forest':<25} | {mae_rf:>10.2f} | {100 - mape_rf:>11.2f}%")
    print(f"{'k-Nearest Neighbors':<25} | {mae_knn:>10.2f} | {100 - mape_knn:>11.2f}%")
    print(f"{'CART (Decision Tree)':<25} | {mae_cart:>10.2f} | {100 - mape_cart:>11.2f}%")
    print("-" * 55)
    print(f"{'FINAL HYBRID ENSEMBLE':<25} | {mae_final:>10.2f} | {100 - mape_final:>11.2f}%")
    print("=" * 55 + "\n")


def train_and_save_models(processed_csv):
    print("No saved models found. Training started...")
    train_df = pd.read_csv(processed_csv).dropna()
    X = train_df.drop('Price', axis=1).values.tolist()
    y = train_df['Price'].values.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestRegressor(n_trees=15, max_depth=10)
    rf.fit(X_train, y_train)

    knn = KNN(k=15)
    knn.fit(X_train, y_train)

    with open(RF_PATH, 'wb') as f: pickle.dump(rf, f)
    with open(KNN_PATH, 'wb') as f: pickle.dump(knn, f)

    run_detailed_validation(rf, knn, processed_csv)
    return rf, knn


def load_trained_models(processed_csv):
    if os.path.exists(RF_PATH) and os.path.exists(KNN_PATH):
        print("Loading pre-trained models from disk...")
        with open(RF_PATH, 'rb') as f:
            rf = pickle.load(f)
        with open(KNN_PATH, 'rb') as f:
            knn = pickle.load(f)

        run_detailed_validation(rf, knn, processed_csv)
        return rf, knn
    else:
        return train_and_save_models(processed_csv)