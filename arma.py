import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import ttest_ind

# Load your data from Excel file
excel_file_path = 'C:/fi/TLV_0_10.xlsx'  # Adjust to your actual file path
ta35_data_excel = pd.read_excel(excel_file_path)

# Assuming the Excel file structure is similar to the CSV, adjust column names as necessary
# Convert the 'date' column (or its equivalent in Hebrew or English) to datetime format and set it as index
ta35_data_excel['date'] = pd.to_datetime(ta35_data_excel['date'], dayfirst=True)  # Adjust 'תאריך' if the column name is different
ta35_data_excel.set_index('date', inplace=True)

# Calculate daily returns for the Excel data
ta35_data_excel['daily_return'] = ta35_data_excel['locking'].pct_change() * 100  # Adjust 'מדד נעילה' if the column name is different
ta35_data_excel.dropna(subset=['daily_return'], inplace=True)

# Fit the ARIMA model to the Excel data (adjust p, d, q as needed)
# We're using the previously determined optimal parameters or you can adjust them based on the data
p, d, q = 4, 0, 4  # Example values for ARIMA(p,d,q), adjust based on your analysis
model_arma_excel = ARIMA(ta35_data_excel['daily_return'], order=(p, d, q))
results_arma_excel = model_arma_excel.fit()

# Store the ARIMA predictions for the Excel data
ta35_data_excel['arma_pred'] = results_arma_excel.predict(start=0, end=len(ta35_data_excel)-1)

# Buy and Hold Strategy for Excel data
initial_investment_excel = 100
cumulative_return_bh_excel = (ta35_data_excel['daily_return'] / 100 + 1).cumprod()

# ARIMA Model-Based Strategy for Excel data
arma_signals_excel = np.where(ta35_data_excel['arma_pred'] > 0, 1, -1)
daily_return_arma_excel = ta35_data_excel['daily_return'] * arma_signals_excel
cumulative_return_arma_excel = (daily_return_arma_excel / 100 + 1).cumprod()

# Conducting a t-test to compare the returns of the two strategies for Excel data
t_stat_excel, p_value_excel = ttest_ind(cumulative_return_bh_excel, cumulative_return_arma_excel)
print("T-statistic for Excel data:", t_stat_excel)
print("P-value for Excel data:", p_value_excel)

# Plotting the development of profits for Excel data
plt.figure(figsize=(15, 8))
plt.plot(ta35_data_excel.index, initial_investment_excel * cumulative_return_bh_excel, label='Buy and Hold Strategy', alpha=0.7)
plt.plot(ta35_data_excel.index, initial_investment_excel * cumulative_return_arma_excel, label='ARIMA Model-Based Strategy', alpha=0.7)
plt.title('Comparison of Investment Strategies: Buy and Hold vs ARIMA Model-Based (Excel Data)')
plt.xlabel('Date')
plt.ylabel('Investment Value (NIS)')
plt.legend()
plt.grid(True)
plt.show()
