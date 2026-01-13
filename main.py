from data import load_data, split_data
from preprocessing import build_preprocessor
from model import train_ridge_cv, save_model, load_model
from evaluation import evaluate_model
from visualization import plot_actual_vs_predicted, plot_feature_vs_price

CSV_PATH = "CarPrice_Assignment.csv"
DEGREE = 2
ALPHA = 1.0

def main():
    # Load and split data
    X, y = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Visualize feature relationships
    for feature in ['enginesize', 'horsepower', 'curbweight', 'citympg']:
        plot_feature_vs_price(X, y, feature)

    # Preprocessing
    preprocessor = build_preprocessor()

    # Train Ridge Polynomial with CV
    model, cv_scores = train_ridge_cv(X_train, y_train, preprocessor, DEGREE, ALPHA)
    print("CV R² scores:", cv_scores)
    print("Mean CV R²:", cv_scores.mean())

    # Test evaluation
    r2, rmse, y_pred = evaluate_model(model, X_test, y_test)
    print(f"Test R²: {r2:.3f}")
    print(f"Test RMSE: {rmse:.2f}")

    # Save & load model
    save_model(model)
    loaded_model = load_model()

    # Visualize predictions
    plot_actual_vs_predicted(y_test, y_pred)

if __name__ == "__main__":
    main()