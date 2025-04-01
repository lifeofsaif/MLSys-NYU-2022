import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def main():
    housing_data = pd.read_csv("./data/train.csv");
    column_names = [
        "1stFlrSF",
        "2ndFlrSF",
        "TotalBsmtSF",
        "LotArea",
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "GarageArea",
    ]

    metrics = []
    for index, column_name in enumerate(column_names):
        columns_to_model = column_names[:index+1]
        
        metricsI = model_columns(housing_data, columns_to_model)
        metrics.append(metricsI)
        
    plot_metrics(housing_data, metrics)

def plot_metrics(housing_data, metrics):
    r2_y =  list(map(lambda x: x['R2'], metrics))
    mse_y =  list(map(lambda x: x['MSE'], metrics))
    mape_y =  list(map(lambda x: x['MAPE'], metrics))
    mae_y =  list(map(lambda x: x['MAE'], metrics))
    

    x = list(range(1, len(metrics) + 1))

    fig, axs = plt.subplots(2, 2) 
    
    axs[0][0].plot(x, mse_y, label='MSE')
    axs[0][0].set_title('Mean Squared Error')
    axs[0][0].set_xticks(x)

    axs[0][1].plot(x, r2_y, label='R2')
    axs[0][1].set_title('R2 Score')
    axs[0][1].set_xticks(x)
    
    axs[1][0].plot(x, mape_y, label='MAPE')
    axs[1][0].set_title('Mean Absolute Percentage Error')
    axs[1][0].set_xticks(x)
    axs[1][0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    axs[1][1].plot(x, mae_y, label='MAE')
    axs[1][1].set_title('Mean Absolute Error')
    axs[1][1].set_xticks(x)
    

    fig.tight_layout()
    fig.savefig("my_plot.png", dpi=300)

def model_columns(housing_data, columns_to_model):
    model = LinearRegression(fit_intercept=True)
    X = housing_data[columns_to_model]
    y = housing_data["SalePrice"]

    model.fit(X, y)
    predictions = model.predict(X)
    
    R2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)

    return {
        "R2": R2,
        "MSE": mse,
        "MAE": mae,
        "MAPE": mape
    }
    
main()