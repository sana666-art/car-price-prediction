from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import joblib
import os

def train_ridge(X_train, y_train, degree=2, alpha=1.0):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)

    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, y_train)

    return model, poly


def train_lasso(X_train, y_train, degree=2, alpha=0.01):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)

    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, y_train)

    return model, poly


def train_ridge_cv(X_train, y_train, preprocessor, degree=2, alpha=1.0):
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('poly', PolynomialFeatures(degree=degree)),
        ('ridge', Ridge(alpha=alpha))
    ])

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring='r2'
    )

    pipeline.fit(X_train, y_train)

    return pipeline, cv_scores

def save_model(model, poly, path="models/ridge_model.pkl"):
    os.makedirs("models", exist_ok=True)
    joblib.dump((model, poly), path)


def load_model(path="models/ridge_model.pkl"):
    return joblib.load(path)