from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def getEvalMetrics(feature_data : tuple[str, np.array(float), np.array(float)]):

    # Calculate Metrics
    y_actual = feature_data[1]
    y_predicted = feature_data[2]
    mae = mean_absolute_error(y_actual, y_predicted)
    mse = mean_squared_error(y_actual, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_predicted)

    return feature_data[0], mae, mse, rmse, r2

def printEvalMetrics( feature_name, mae, mse, rmse, r2):

    print(f"{'Metric':<30}{'Value':>10} for {feature_name}")
    print("-" * 65)
    print(f"{'Mean Absolute Error (MAE)':<30}{mae:>10.4f}")
    print(f"{'Mean Squared Error (MSE)':<30}{mse:>10.4f}")
    print(f"{'Root Mean Squared Error (RMSE)':<30}{rmse:>10.4f}")
    print(f"{'R-squared (R^2)':<30}{r2:>10.4f}")
    print(f"\n\n")
