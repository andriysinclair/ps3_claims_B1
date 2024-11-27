# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score
# %%
def evaluate_predictions(y_true, y_pred, sample_weight=None):
    metrics = {}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Bias
    bias = np.mean(y_pred - y_true)

    # Deviance
    deviance = np.sum((y_true - y_pred) ** 2)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Lorenz curve
    sorted_idx = np.argsort(y_pred)
    sorted_true = y_true[sorted_idx]
    sorted_pred = y_pred[sorted_idx]
    
    cum_true = np.cumsum(sorted_true) / np.sum(sorted_true)
    cum_pred = np.cumsum(sorted_pred) / np.sum(sorted_pred)
    
    # Gini coefficient
    lorenz_curve_area = np.trapz(cum_true, cum_pred)
    gini_coefficient = 2 * lorenz_curve_area - 1

    metrics = {
        'Bias': bias,
        'Deviance': deviance,
        'MAE': mae,
        'RMSE': rmse
        'Gini': gini_coefficient
    }

    # Return the metrics
    return pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).set_index('Metric')