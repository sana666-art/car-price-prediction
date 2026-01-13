from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on test data.

    Works for:
    - sklearn Pipeline (with preprocessing & polynomial features)
    - Separate polynomial + linear models

    Returns:
        r2  : R² score
        rmse: Root Mean Squared Error
        y_pred: Predictions
    """
    try:
        # If model is a Pipeline
        y_pred = model.predict(X_test)
    except:
        # If model and poly are separate (tuple)
        model_obj, poly = model
        X_test_poly = poly.transform(X_test)
        y_pred = model_obj.predict(X_test_poly)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, rmse, y_pred
