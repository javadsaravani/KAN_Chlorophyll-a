import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import torch.nn as nn
from scipy.stats import pearsonr
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from deepkan import SplineLinearLayer
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lag=12
def prepare_data(raw_data):
    raw_data['date'] = pd.to_datetime(raw_data['date'], errors='coerce')

    start_date = raw_data['date'].min()
    end_date = raw_data['date'].max()
    chl_a_values = raw_data[['chl_a']].values
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    raw_data['lof_score'] = lof.fit_predict(chl_a_values)
    raw_data = raw_data[raw_data['lof_score'] != -1].drop(columns=['lof_score'])
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    raw_data.set_index('date', inplace=True)
    raw_data = raw_data.reindex(date_range)
    missing_dates = raw_data[raw_data['chl_a'].isnull()].index
    raw_data['chl_a'] = raw_data['chl_a'].fillna(method='ffill')
    raw_data['chl_a'] = raw_data['chl_a'].fillna(method='bfill')
    stl = STL(raw_data['chl_a'], seasonal=13)
    result = stl.fit()
    raw_data['chl_a'] = np.where(raw_data['chl_a'].isnull(), result.trend + result.seasonal, raw_data['chl_a'])
    raw_data['chl_a'] = raw_data['chl_a'].interpolate(method='linear')
    raw_data['month'] = raw_data.index.month
    print(f"Data range from {start_date} to {end_date}")
    raw_data['month_sin'] = np.sin(2 * np.pi * raw_data['month'] / 12)
    raw_data['month_cos'] = np.cos(2 * np.pi * raw_data['month'] / 12)
    raw_data = raw_data.drop(['month'], axis=1)
    data_LSTM = raw_data.copy()
        
    # Create lag features
    for i in range(1, lag +1):
        raw_data[f'yt-{i}'] = raw_data['chl_a'].shift(i)
    
    raw_data.dropna(inplace=True)
    
    return raw_data, data_LSTM
    
class KANTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_knots=5, spline_order=3,
                 noise_scale=0.1, base_scale=0.1, spline_scale=1.0,
                 activation=nn.SiLU, grid_epsilon=1, grid_range=[-1, 1]):
        super(KANTimeSeries, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(SplineLinearLayer(prev_size, hidden_size, num_knots, spline_order,
                                                 noise_scale, base_scale, spline_scale,
                                                 activation, grid_epsilon, grid_range))
            prev_size = hidden_size

        self.output_layer = SplineLinearLayer(prev_size, output_size, num_knots, spline_order,
                                              noise_scale, base_scale, spline_scale,
                                              activation, grid_epsilon, grid_range)

    def forward(self, x, update_knots=False):
        for layer in self.layers:
            if update_knots:
                layer._update_knots(x)
            x = layer(x)

        if update_knots:
            self.output_layer._update_knots(x)
        x = self.output_layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        loss = 0
        for layer in self.layers:
            loss += layer._regularization_loss(regularize_activation, regularize_entropy)
        loss += self.output_layer._regularization_loss(regularize_activation, regularize_entropy)
        return loss

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        loss = 0
        for layer in self.layers:
            loss += layer._regularization_loss(regularize_activation, regularize_entropy)
        loss += self.output_layer._regularization_loss(regularize_activation, regularize_entropy)
        return loss


def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return
    try:
        global raw_data
        raw_data = pd.read_csv(file_path)
        messagebox.showinfo("File Loaded", "Make sure your data contains 'date' and 'chl_a' columns")
    except Exception as e:
        messagebox.showerror("Error", f"Could not load file: {e}")


def next_step():
    if raw_data is None:
        messagebox.showwarning("Warning", "Please load a CSV file first.")
        return
    
    global data, data_LSTM
    data, data_LSTM = prepare_data(raw_data) 
    
    for widget in frame.winfo_children():
        widget.destroy()
    
    label = tk.Label(frame, text="Select a Model:")
    label.pack(pady=10)
    
    models = ["KAN", "MLP-NN", "LSTM", "GRU", "RF", "GPR", "SVR"]
    model_var.set(models[0])
    
    dropdown = tk.OptionMenu(frame, model_var, *models)
    dropdown.pack(pady=10)
    
    next_button = tk.Button(frame, text="Run Model", command=run_model)
    next_button.pack(pady=10)
    


def plot_actual_vs_predicted(actual, predicted, model=None, data=None, forecast_steps=6, selected_model=None):

    for widget in frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(actual, label='Actual', color='blue')
    ax.plot(predicted, label='Predicted', color='red', linestyle='--')
    ax.set_title('Actual vs. Predicted - Test Data')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chl-a')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    plt.close(fig)

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    metrics_text = f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}"
    metrics_label = tk.Label(frame, text=metrics_text)
    metrics_label.pack(pady=10)

    if model is not None and data is not None:
        forecast_button = None  
        if selected_model == 'MLP-NN':
            forecast_button = tk.Button(frame, text="Forecast", command=lambda: MLP_forecast_and_plot(model, data, actual, predicted, forecast_steps))
        elif selected_model == 'KAN':
            forecast_button = tk.Button(frame, text="Forecast", command=lambda: KAN_forecast_and_plot(model, data, actual, predicted, forecast_steps))
        elif selected_model == 'LSTM':
            forecast_button = tk.Button(frame, text="Forecast", command=lambda: LSTM_forecast_and_plot(model, data, actual, predicted, forecast_steps))
        elif selected_model == 'GRU':
            forecast_button = tk.Button(frame, text="Forecast", command=lambda: GRU_forecast_and_plot(model, data, actual, predicted, forecast_steps))
        elif selected_model == 'RF':  
            forecast_button = tk.Button(frame, text="Forecast", command=lambda: RF_forecast_and_plot(model, data, actual, predicted, forecast_steps))
        elif selected_model == 'GPR':
            forecast_button = tk.Button(frame, text="Forecast", command=lambda: GPR_forecast_and_plot(model, data, actual, predicted, forecast_steps))
        elif selected_model == 'SVR':
            forecast_button = tk.Button(frame, text="Forecast", command=lambda: SVR_forecast_and_plot(model, data, actual, predicted, forecast_steps))


        if forecast_button is not None:
            forecast_button.pack(pady=10)

    return_button = tk.Button(frame, text="Return", command=next_step)
    return_button.pack(pady=10)

def calculate_month_cyclic_features(date):
    month = date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return month_sin, month_cos

def prepare_forecast_input(last_lag_values, forecast_dates, i):
    month_sin, month_cos = calculate_month_cyclic_features(forecast_dates[i])

    forecast_input = np.hstack([last_lag_values, month_sin, month_cos])

    return forecast_input

def MLP_forecast_next_records(model, data, lag=12, forecast_steps=6):
    last_lag_values = data['chl_a'].values[-lag:].flatten()

  
    last_date = data.index[-1]
    forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_steps + 1)]

    forecast_values = []
    scaler_y = StandardScaler()
    scaler_y.fit(data[['chl_a']].values)


    for i in range(forecast_steps):
 
        month_sin, month_cos = calculate_month_cyclic_features(forecast_dates[i])

        forecast_input = np.hstack([last_lag_values, month_sin, month_cos])
        forecast_input_tensor = torch.tensor(forecast_input, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            forecast_value_scaled = model(forecast_input_tensor).cpu().numpy().flatten()[0]

        forecast_value = scaler_y.inverse_transform(np.array([[forecast_value_scaled]])).flatten()[0]

        forecast_values.append(forecast_value)

        last_lag_values = np.roll(last_lag_values, -1)  
        last_lag_values[-1] = forecast_value 

    return forecast_values, forecast_dates

def MLP_forecast_and_plot(model, data, actual, predicted, forecast_steps=6):
    forecast_values, forecast_dates = MLP_forecast_next_records(model, data, lag=lag, forecast_steps=forecast_steps)

    extended_actual = np.append(actual, [None] * forecast_steps)  
    extended_predicted = np.append(predicted, forecast_values) 

    for widget in frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(extended_actual, label='Actual', color='blue')
    ax.plot(extended_predicted[:len(predicted)], label='Predicted', color='red', linestyle='--')
    ax.plot(range(len(predicted), len(extended_predicted)), extended_predicted[len(predicted):], label='Forecast', color='green', linestyle='--')
    ax.set_title('Actual vs. Predicted - Including Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chl-a')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    plt.close(fig)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    metrics_text = f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}"
    metrics_label = tk.Label(frame, text=metrics_text)
    metrics_label.pack(pady=10)

    return_button = tk.Button(frame, text="Return", command=next_step)
    return_button.pack(pady=10)

def KAN_forecast_next_records(model, last_window, n_steps, window_size, start_date):
    forecasts = []
    current_window = last_window.copy().values  
    current_date = start_date

    for _ in range(n_steps):
      
        current_window_torch = torch.tensor(current_window.reshape(1, window_size * 3), dtype=torch.float32)

   
        with torch.no_grad():
            next_pred = model(current_window_torch).item()  

   
        forecasts.append(next_pred)

    
        current_date += pd.DateOffset(months=1)  
        month_sin, month_cos = calculate_month_cyclic_features(current_date)

    
        next_input = np.array([next_pred, month_sin, month_cos]) 
        current_window = np.vstack((current_window[1:], next_input))  

    return forecasts


def KAN_forecast_and_plot(model, data, actual, predicted, forecast_steps=6):
    last_window = data[-12:] 
    start_date = pd.to_datetime(data.index[-1])
    
    forecast_values = KAN_forecast_next_records(model, last_window, forecast_steps, window_size=12, start_date=start_date)

    extended_actual = np.append(actual, [None] * forecast_steps) 
    extended_predicted = np.append(predicted, forecast_values) 

    for widget in frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(extended_actual, label='Actual', color='blue')
    ax.plot(extended_predicted[:len(predicted)], label='Predicted', color='red', linestyle='--')
    ax.plot(range(len(predicted), len(extended_predicted)), extended_predicted[len(predicted):], label='Forecast', color='green', linestyle='--')
    ax.set_title('Actual vs. Predicted - Including Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chl-a')
    ax.legend()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    plt.close(fig)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    metrics_text = f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}"
    metrics_label = tk.Label(frame, text=metrics_text)
    metrics_label.pack(pady=10)

    return_button = tk.Button(frame, text="Return", command=next_step)
    return_button.pack(pady=10)

def LSTM_forecast_next_records(model, data, time_steps=12, forecast_steps=6):
    last_window = data['chl_a'].values[-time_steps:].reshape(-1, 1)
    month_sin = data['month_sin'].values[-time_steps:]
    month_cos = data['month_cos'].values[-time_steps:]

    last_window = np.hstack([last_window, month_sin.reshape(-1, 1), month_cos.reshape(-1, 1)])
    
    forecasts = []
    
    model_input = torch.tensor(last_window.reshape(1, time_steps, 3), dtype=torch.float32).to(device)
    
    for _ in range(forecast_steps):
        model.eval()
        with torch.no_grad():
            forecast_value_scaled = model(model_input).cpu().numpy().flatten()[0]
        
        scaler_y = StandardScaler()
        scaler_y.fit(data[['chl_a']].values)
        forecast_value = scaler_y.inverse_transform(np.array([[forecast_value_scaled]])).flatten()[0]

        forecasts.append(forecast_value)

        next_month_sin = np.sin(2 * np.pi * ((month_sin[-1] + 1) % 12) / 12)
        next_month_cos = np.cos(2 * np.pi * ((month_cos[-1] + 1) % 12) / 12)
        next_input = np.array([forecast_value, next_month_sin, next_month_cos]).reshape(1, 1, -1)
        
        model_input = torch.cat((model_input[:, 1:, :], torch.tensor(next_input, dtype=torch.float32).to(device)), dim=1)

    return forecasts

def LSTM_forecast_and_plot(model, data, actual, predicted, forecast_steps=6):
    forecast_values = LSTM_forecast_next_records(model, data, time_steps=lag, forecast_steps=forecast_steps)
    
    extended_actual = np.append(actual, [None] * forecast_steps) 
    extended_predicted = np.append(predicted, forecast_values) 

    for widget in frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(extended_actual, label='Actual', color='blue')
    ax.plot(extended_predicted[:len(predicted)], label='Predicted', color='red', linestyle='--')
    ax.plot(range(len(predicted), len(extended_predicted)), extended_predicted[len(predicted):], label='Forecast', color='green', linestyle='--')
    ax.set_title('Actual vs. Predicted - Including Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chl-a')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    plt.close(fig)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    metrics_text = f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}"
    metrics_label = tk.Label(frame, text=metrics_text)
    metrics_label.pack(pady=10)

    return_button = tk.Button(frame, text="Return", command=next_step)
    return_button.pack(pady=10)

def prepare_data_with_time_steps(data_LSTM, time_steps=12):
    chl_a_values = data_LSTM['chl_a'].values
    month_sin = data_LSTM['month_sin'].values
    month_cos = data_LSTM['month_cos'].values

    X, y = [], []
    for i in range(len(chl_a_values) - time_steps):
        X.append(
               np.hstack([chl_a_values[i:i + time_steps].reshape(-1, 1), 
               np.tile([month_sin[i + time_steps], month_cos[i + time_steps]], (time_steps, 1))])
        )

        y.append(chl_a_values[i + time_steps])  

    X, y = np.array(X), np.array(y).reshape(-1, 1)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = X.reshape(X.shape[0], -1) 
    X = scaler_X.fit_transform(X) 
    X = X.reshape(X.shape[0], time_steps, -1)  
    y = scaler_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y

def GRU_forecast_next_records(model, data, time_steps=12, forecast_steps=6):
    last_window = data['chl_a'].values[-time_steps:].reshape(-1, 1)
    month_sin = data['month_sin'].values[-time_steps:]
    month_cos = data['month_cos'].values[-time_steps:]
    last_window = np.hstack([last_window, month_sin.reshape(-1, 1), month_cos.reshape(-1, 1)])
    forecasts = []
    

    model_input = torch.tensor(last_window.reshape(1, time_steps, 3), dtype=torch.float32).to(device)
    
    for _ in range(forecast_steps):
        model.eval()
        with torch.no_grad():
            forecast_value_scaled = model(model_input).cpu().numpy().flatten()[0]
        
        scaler_y = StandardScaler()
        scaler_y.fit(data[['chl_a']].values)
        forecast_value = scaler_y.inverse_transform(np.array([[forecast_value_scaled]])).flatten()[0]

        forecasts.append(forecast_value)

        next_month_sin = np.sin(2 * np.pi * ((month_sin[-1] + 1) % 12) / 12)
        next_month_cos = np.cos(2 * np.pi * ((month_cos[-1] + 1) % 12) / 12)
        next_input = np.array([forecast_value, next_month_sin, next_month_cos]).reshape(1, 1, -1)
        
        model_input = torch.cat((model_input[:, 1:, :], torch.tensor(next_input, dtype=torch.float32).to(device)), dim=1)

    return forecasts

def GRU_forecast_and_plot(model, data, actual, predicted, forecast_steps=6):
    forecast_values = GRU_forecast_next_records(model, data, time_steps=lag, forecast_steps=forecast_steps)

    extended_actual = np.append(actual, [None] * forecast_steps)
    extended_predicted = np.append(predicted, forecast_values)  
    for widget in frame.winfo_children():
        widget.destroy()

  
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(extended_actual, label='Actual', color='blue')
    ax.plot(extended_predicted[:len(predicted)], label='Predicted', color='red', linestyle='--')
    ax.plot(range(len(predicted), len(extended_predicted)), extended_predicted[len(predicted):], label='Forecast', color='green', linestyle='--')
    ax.set_title('Actual vs. Predicted - Including Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chl-a')
    ax.legend()

 
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

   
    plt.close(fig)
      
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    metrics_text = f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}"
    metrics_label = tk.Label(frame, text=metrics_text)
    metrics_label.pack(pady=10)

 
    return_button = tk.Button(frame, text="Return", command=next_step)
    return_button.pack(pady=10)



def RF_forecast_next_records(model, last_lag_values, n_steps, lag, forecast_dates):
    forecasts = []
    current_lag_values = last_lag_values.copy()

    for i in range(n_steps):
       
        forecast_input = np.hstack([current_lag_values, forecast_dates[i][0], forecast_dates[i][1]])
        forecast_value_scaled = model.predict(forecast_input.reshape(1, -1)).flatten()[0]
        forecasts.append(forecast_value_scaled)
        current_lag_values = np.roll(current_lag_values, -1)
        current_lag_values[-1] = forecast_value_scaled

    return forecasts


def RF_forecast_and_plot(model, data, actual, predicted, forecast_steps=6):
    print("RF forecast and plot function called.")
    last_lag_values = data[['chl_a']].values[-lag:].flatten()
    last_date = data.index[-1]
    forecast_dates = [calculate_month_cyclic_features(last_date + pd.DateOffset(months=i)) for i in range(1, forecast_steps + 1)]

   
    next_6_predictions_rf = RF_forecast_next_records(model, last_lag_values, forecast_steps, lag, forecast_dates)

   
    extended_actual = np.append(actual, [None] * forecast_steps) 
    extended_predicted = np.append(predicted, next_6_predictions_rf)  

  
    for widget in frame.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(extended_actual, label='Actual', color='blue')
    ax.plot(extended_predicted[:len(predicted)], label='Predicted', color='red', linestyle='--')
    ax.plot(range(len(predicted), len(extended_predicted)), extended_predicted[len(predicted):], label='Forecast', color='green', linestyle='--')
    ax.set_title('Actual vs. Predicted - Including Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chl-a')
    ax.legend()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    plt.close(fig)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    metrics_text = f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}"
    metrics_label = tk.Label(frame, text=metrics_text)
    metrics_label.pack(pady=10)
    return_button = tk.Button(frame, text="Return", command=next_step)
    return_button.pack(pady=10)


def GPR_forecast_next_records(model, last_lag_values, n_steps, lag, forecast_dates):
    forecasts = []
    current_lag_values = last_lag_values.copy()

    for i in range(n_steps):
        forecast_input = np.hstack([current_lag_values, forecast_dates[i][0], forecast_dates[i][1]])
        forecast_value_scaled = model.predict(forecast_input.reshape(1, -1)).flatten()[0]
        forecasts.append(forecast_value_scaled)
        current_lag_values = np.roll(current_lag_values, -1)
        current_lag_values[-1] = forecast_value_scaled

    return forecasts




def GPR_forecast_and_plot(model, data, actual, predicted, forecast_steps=6):
    print("GPR forecast and plot function called.")
    last_lag_values = data[['chl_a']].values[-lag:].flatten()
    last_date = data.index[-1]
    forecast_dates = [calculate_month_cyclic_features(last_date + pd.DateOffset(months=i)) for i in range(1, forecast_steps + 1)]

    next_6_predictions_rf = GPR_forecast_next_records(model, last_lag_values, forecast_steps, lag, forecast_dates)

    extended_actual = np.append(actual, [None] * forecast_steps)  
    extended_predicted = np.append(predicted, next_6_predictions_rf)  

    for widget in frame.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(extended_actual, label='Actual', color='blue')
    ax.plot(extended_predicted[:len(predicted)], label='Predicted', color='red', linestyle='--')
    ax.plot(range(len(predicted), len(extended_predicted)), extended_predicted[len(predicted):], label='Forecast', color='green', linestyle='--')
    ax.set_title('Actual vs. Predicted - Including Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chl-a')
    ax.legend()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    plt.close(fig)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    metrics_text = f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}"
    metrics_label = tk.Label(frame, text=metrics_text)
    metrics_label.pack(pady=10)
    return_button = tk.Button(frame, text="Return", command=next_step)
    return_button.pack(pady=10)


def SVR_forecast_next_records(model, last_lag_values, n_steps, lag, forecast_dates):
    forecasts = []
    current_lag_values = last_lag_values.copy()

    for i in range(n_steps):
        forecast_input = np.hstack([current_lag_values, forecast_dates[i][0], forecast_dates[i][1]])

        forecast_value_scaled = model.predict(forecast_input.reshape(1, -1)).flatten()[0]
        forecasts.append(forecast_value_scaled)
        current_lag_values = np.roll(current_lag_values, -1)
        current_lag_values[-1] = forecast_value_scaled

    return forecasts




def SVR_forecast_and_plot(model, data, actual, predicted, forecast_steps=6):

    print("SVR forecast and plot function called.")
    last_lag_values = data[['chl_a']].values[-lag:].flatten()
    last_date = data.index[-1]
    forecast_dates = [calculate_month_cyclic_features(last_date + pd.DateOffset(months=i)) for i in range(1, forecast_steps + 1)]


    next_6_predictions_rf = SVR_forecast_next_records(model, last_lag_values, forecast_steps, lag, forecast_dates)


    extended_actual = np.append(actual, [None] * forecast_steps) 
    extended_predicted = np.append(predicted, next_6_predictions_rf) 


    for widget in frame.winfo_children():
        widget.destroy()


    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(extended_actual, label='Actual', color='blue')
    ax.plot(extended_predicted[:len(predicted)], label='Predicted', color='red', linestyle='--')
    ax.plot(range(len(predicted), len(extended_predicted)), extended_predicted[len(predicted):], label='Forecast', color='green', linestyle='--')
    ax.set_title('Actual vs. Predicted - Including Forecast')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chl-a')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()


    plt.close(fig)

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    metrics_text = f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}"
    metrics_label = tk.Label(frame, text=metrics_text)
    metrics_label.pack(pady=10)

    return_button = tk.Button(frame, text="Return", command=next_step)
    return_button.pack(pady=10)


def run_model():
    selected_model = model_var.get()
    lag=12

    if selected_model in ['MLP-NN', 'RF', 'GPR', 'SVR']:

        X = data[[f'yt-{i}' for i in range(1, lag + 1)] + ['month_sin', 'month_cos']].values
        y = data['chl_a'].values

    elif selected_model in ['LSTM', 'GRU', 'KAN']:

        X = data_LSTM[['chl_a', 'month_sin', 'month_cos']].values
        y = data_LSTM['chl_a'].values

    else:
        messagebox.showerror("Error", "Model not supported.")
        return
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    

    if selected_model == 'MLP-NN':

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        class ANNModel(nn.Module):
            def __init__(self):
                super(ANNModel, self).__init__()
                self.fc1 = nn.Linear(14, 64) 
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(64, 1) 

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        def train_and_evaluate_ann():
            model = ANNModel()

          
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  
            num_epochs = 50
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()

                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)

                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                predictions_train = model(X_train_tensor).numpy()
                predictions_test = model(X_test_tensor).numpy()
                actual_train = y_train_tensor.numpy()
                actual_test = y_test_tensor.numpy()

            return model, predictions_train, actual_train, predictions_test, actual_test
        
        model, predictions_train, actual_train, predictions_test, actual_test = train_and_evaluate_ann()
        plot_actual_vs_predicted(actual_test.flatten(), predictions_test.flatten(), model=model, data=data, forecast_steps=6, selected_model='MLP-NN')
   
    elif selected_model == 'KAN':

        dat_KAN = data_LSTM[['chl_a', 'month_sin', 'month_cos']].values.astype(np.float32)

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = scaler.fit_transform(dat_KAN)

        def create_sequences(dat_KAN, window_size):
            sequences = []
            for i in range(len(dat_KAN) - window_size):
                input_seq = dat_KAN[i:i + window_size]
                output = dat_KAN[i + window_size, 0] 
                sequences.append((input_seq, output))
            return sequences

        window_size = 12
        sequences = create_sequences(data_normalized, window_size)
        inputs_torch = torch.tensor([seq[0] for seq in sequences], dtype=torch.float32)
        targets_torch = torch.tensor([seq[1] for seq in sequences], dtype=torch.float32)

        split_idx = int(len(inputs_torch) * 0.8)
        train_inputs_torch = inputs_torch[:split_idx]
        test_inputs_torch = inputs_torch[split_idx:]
        train_targets_torch = targets_torch[:split_idx]
        test_targets_torch = targets_torch[split_idx:]

        input_size = window_size * 3
        hidden_sizes = [32]
        num_knots = 5
        spline_order = 3
        model_torch = KANTimeSeries(input_size, hidden_sizes, 1, num_knots, spline_order)

        optimizer_torch = torch.optim.Adam(model_torch.parameters(), lr=0.01)
        criterion_torch = torch.nn.MSELoss()

        epochs = 10
        for epoch in range(epochs):
            model_torch.train()
            epoch_loss = 0
            for i in range(len(train_inputs_torch)):
                optimizer_torch.zero_grad()
                output = model_torch(train_inputs_torch[i:i + 1].view(1, -1))
                target = train_targets_torch[i].view_as(output)
                loss = criterion_torch(output, target)
                loss.backward()
                optimizer_torch.step()
                epoch_loss += loss.item()

        model_torch.eval()
        with torch.no_grad():
            train_pred_torch = model_torch(train_inputs_torch.view(len(train_inputs_torch), -1)).squeeze().numpy()
            test_pred_torch = model_torch(test_inputs_torch.view(len(test_inputs_torch), -1)).squeeze().numpy()
        test_pred_denorm_torch = scaler.inverse_transform(np.hstack([test_pred_torch.reshape(-1, 1), np.zeros((len(test_pred_torch), 2))]))[:, 0]
        test_actual_denorm_torch = scaler.inverse_transform(np.hstack([test_targets_torch.numpy().reshape(-1, 1), np.zeros((len(test_targets_torch), 2))]))[:, 0]

        plot_actual_vs_predicted(test_actual_denorm_torch, test_pred_denorm_torch, model=model_torch, data=data_LSTM, forecast_steps=6, selected_model='KAN')


    elif selected_model == 'LSTM':
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                lstm_out = lstm_out[:, -1, :]
                out = self.fc(lstm_out)
                return out
        def train_and_evaluate_lstm(X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, num_epochs=100, lr=0.01):
            lstm_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

            for epoch in range(num_epochs):
                lstm_model.train()
                optimizer.zero_grad()
                output = lstm_model(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()
        
            lstm_model.eval()
            with torch.no_grad():
                y_train_pred = lstm_model(X_train).cpu().numpy()
                y_test_pred = lstm_model(X_test).cpu().numpy()
        
            return y_train_pred, y_test_pred, y_train.cpu().numpy(), y_test.cpu().numpy(), lstm_model
        time_steps = lag 
        input_size = 3 
        hidden_size = 32
        output_size = 1
    
        X_train, y_train, X_test, y_test, scaler_y = prepare_data_with_time_steps(data_LSTM, time_steps=time_steps)
    
        y_train_pred, y_test_pred, y_train_actual, y_test_actual, lstm_model = train_and_evaluate_lstm(
            X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, num_epochs=100, lr=0.01
        )
    
        y_train_pred = scaler_y.inverse_transform(y_train_pred)
        y_test_pred = scaler_y.inverse_transform(y_test_pred)
        y_train_actual = scaler_y.inverse_transform(y_train_actual)
        y_test_actual = scaler_y.inverse_transform(y_test_actual)
    
        y_train_pred = y_train_pred.flatten()
        y_test_pred = y_test_pred.flatten()
        y_train_actual = y_train_actual.flatten()
        y_test_actual = y_test_actual.flatten()
    
        plot_actual_vs_predicted(
            y_test_actual,
            y_test_pred,
            model=lstm_model,
            data=data_LSTM,
            forecast_steps=6,
            selected_model='LSTM'
        )
    elif selected_model == 'GRU':
        class GRUModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(GRUModel, self).__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                gru_out, _ = self.gru(x)
                gru_out = gru_out[:, -1, :] 
                out = self.fc(gru_out)
                return out

        time_steps = lag 
        input_size = 3 
        hidden_size = 32
        output_size = 1
    
        X_train, y_train, X_test, y_test, scaler_y = prepare_data_with_time_steps(data_LSTM, time_steps=time_steps)
    
        def train_and_evaluate_gru(X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, num_epochs=100, lr=0.001):
            gru_model = GRUModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(gru_model.parameters(), lr=lr)
            
            for epoch in range(num_epochs):
                gru_model.train()
                optimizer.zero_grad()
                output = gru_model(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()
    
            gru_model.eval()
            with torch.no_grad():
                y_train_pred = gru_model(X_train).cpu().numpy()
                y_test_pred = gru_model(X_test).cpu().numpy()
    
            return y_train_pred, y_test_pred, y_train.cpu().numpy(), y_test.cpu().numpy(), gru_model
    
        y_train_pred, y_test_pred, y_train_actual, y_test_actual, gru_model = train_and_evaluate_gru(
            X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, num_epochs=100, lr=0.001
        )
    
        y_train_pred = scaler_y.inverse_transform(y_train_pred)
        y_test_pred = scaler_y.inverse_transform(y_test_pred)
        y_train_actual = scaler_y.inverse_transform(y_train_actual)
        y_test_actual = scaler_y.inverse_transform(y_test_actual)
    
        y_train_pred = y_train_pred.flatten()
        y_test_pred = y_test_pred.flatten()
        y_train_actual = y_train_actual.flatten()
        y_test_actual = y_test_actual.flatten()
    
        plot_actual_vs_predicted(
            y_test_actual,
            y_test_pred,
            model=gru_model,
            data=data_LSTM,
            forecast_steps=6,
            selected_model='GRU'
        )

    elif selected_model == 'RF':
        def prepare_data_with_lag(data):
            X = data[[f'yt-{i}' for i in range(1, lag + 1)] + ['month_sin', 'month_cos']].values
            y = data['chl_a'].values
    
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
    
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
            return X_train, y_train, X_test, y_test, scaler_y
    
        lag = 12  
        X_train, y_train, X_test, y_test, scaler_y = prepare_data_with_lag(data)
    
        def train_and_evaluate_rf(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None):
            rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf_model.fit(X_train, y_train)
    
            y_test_pred = rf_model.predict(X_test)
    
            y_test_pred_denorm = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
            y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
            return y_test_actual, y_test_pred_denorm, rf_model
        actual_values, predicted_values, rf_model = train_and_evaluate_rf(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=5)
    
        plot_actual_vs_predicted(actual_values, predicted_values, model=rf_model, data=data, forecast_steps=6, selected_model='RF')

    elif selected_model == 'GPR':
        def prepare_data_with_lag(data):
            X = data[[f'yt-{i}' for i in range(1, lag + 1)] + ['month_sin', 'month_cos']].values
            y = data['chl_a'].values
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
    
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
            return X_train, y_train, X_test, y_test, scaler_y
    
        lag = 12  
        X_train, y_train, X_test, y_test, scaler_y = prepare_data_with_lag(data)
    
        def train_and_evaluate_gpr(X_train, y_train, X_test, y_test):
            kernel = ExpSineSquared(length_scale=1.0, periodicity=12.0) + WhiteKernel(noise_level=1e-5)
            gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    
            gpr_model.fit(X_train, y_train)
    
            y_test_pred = gpr_model.predict(X_test)
    
            y_test_pred_denorm = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
            y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
            return y_test_actual, y_test_pred_denorm, gpr_model
        actual_values, predicted_values, gpr_model = train_and_evaluate_gpr(X_train, y_train, X_test, y_test)
    
        plot_actual_vs_predicted(actual_values, predicted_values, model=gpr_model, data=data, forecast_steps=6, selected_model='GPR')

    elif selected_model == 'SVR':

        def prepare_data_with_lag(data):
            X = data[[f'yt-{i}' for i in range(1, lag + 1)] + ['month_sin', 'month_cos']].values
            y = data['chl_a'].values

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
    
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
            return X_train, y_train, X_test, y_test, scaler_y

        lag = 12  
        X_train, y_train, X_test, y_test, scaler_y = prepare_data_with_lag(data)
    
        def train_and_evaluate_svr(X_train, y_train, X_test, y_test):
            svr_model = SVR()
            svr_model.fit(X_train, y_train)
            y_test_pred = svr_model.predict(X_test)
    
            y_test_pred_denorm = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
            y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
            return y_test_actual, y_test_pred_denorm, svr_model
    
        actual_values, predicted_values, svr_model = train_and_evaluate_svr(X_train, y_train, X_test, y_test)
        plot_actual_vs_predicted(actual_values, predicted_values, model=svr_model, data=data, forecast_steps=6, selected_model='SVR')

# GUI Setup
root = tk.Tk()
root.title("Chl-a Prediction Model GUI")


frame = tk.Frame(root)
frame.pack(pady=50, padx=50)
raw_data = None
prepared_data = None
model_var = tk.StringVar()
forecast_entry = tk.Entry(frame)
load_button = tk.Button(frame, text="Load CSV", command=load_csv)
load_button.pack(pady=10)
next_button = tk.Button(frame, text="Next", command=next_step)
next_button.pack(pady=10)
root.mainloop()
