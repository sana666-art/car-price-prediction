import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_test, y_pred):
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Car Prices")
    plt.show()

def plot_feature_vs_price(X, y, feature):
    plt.scatter(X[feature], y, alpha=0.7)
    plt.xlabel(feature)
    plt.ylabel("Price")
    plt.title(f"{feature} vs Car Price")
    plt.show()