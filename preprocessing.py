from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_preprocessor():
    """
    Creates a preprocessing pipeline:
    - Imputes missing values for numerical and categorical features
    - OneHotEncodes categorical features
    - Scales numerical features
    """

    # Numerical and categorical columns
    numerical_features = ['enginesize', 'horsepower', 'curbweight', 'citympg']
    categorical_features = ['fueltype', 'aspiration', 'carbody']

    # Pipeline for numerical features: impute missing → scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # replace nulls with mean
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features: impute missing → one-hot encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # replace nulls with mode
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ]
    )

    return preprocessor
