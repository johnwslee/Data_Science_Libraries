# Import packages
import pandas as pd
from scipy.special import inv_boxcox
from pmdarima import auto_arima
import pickle

# Read in clean data
data = pd.read_csv('clean_data.csv')

# Unpickle the lam variable
with open('lam.pickle', 'rb') as f:
    lam = pickle.load(f)

# Split train and test
train = data.iloc[:-int(len(data) * 0.2)]
test = data.iloc[-int(len(data) * 0.2):]

# pmdarima to fit the ARIMA model
model = auto_arima(train['Passengers_Boxcox'], d=None, seasonal=True, m=12, suppress_warnings=True)

# Forecast using the fitted model
boxcox_forecasts = model.predict(n_periods=len(test))
forecasts = inv_boxcox(boxcox_forecasts, lam)

# Save train, test, and forecasts to CSV files
train.to_csv('train_data.csv', index=False)
test.to_csv('test_data.csv', index=False)
forecasts_df = pd.DataFrame({'Forecasts': forecasts}, index=test.index)
forecasts_df.to_csv('forecasts.csv', index=True)