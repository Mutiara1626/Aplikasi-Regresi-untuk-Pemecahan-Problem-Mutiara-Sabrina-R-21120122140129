import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Load data
data = pd.read_csv('/content/Student_Performance.csv')

# Mengambil kolom yang relevan untuk Problem 2
NL = data['Sample Question Papers Practiced'].values.reshape(-1, 1)  # Jumlah Latihan Soal
NT = data['Performance Index'].values  # Nilai Ujian

# Menambah nilai kecil pada NL untuk menghindari masalah log dan power
NL_no_zeros = NL + 1e-10

# Metode 1: Model Linear
model_linear = LinearRegression()
model_linear.fit(NL, NT)
NT_pred_linear = model_linear.predict(NL)

# Metode 2: Model Pangkat Sederhana
def power_law(x, a, b):
    return a * np.power(x, b)

params, _ = curve_fit(power_law, NL_no_zeros.flatten(), NT)
NT_pred_power = power_law(NL_no_zeros, *params)

# Metode Opsional: Model Logaritmik
def logarithmic(x, a, b):
    return a + b * np.log(x)

params_log, _ = curve_fit(logarithmic, NL_no_zeros.flatten(), NT)
NT_pred_log = logarithmic(NL_no_zeros, *params_log)

# Plot grafik titik data dan hasil regresinya masing-masing
plt.figure(figsize=(14, 7))

plt.subplot(1, 3, 1)
plt.scatter(NL, NT, color='blue', label='Data')
plt.plot(NL, NT_pred_linear, color='red', label='Linear Fit')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Model Linear')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(NL, NT, color='blue', label='Data')
plt.plot(NL, NT_pred_power, color='green', label='Power Fit')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Model Pangkat Sederhana')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(NL, NT, color='blue', label='Data')
plt.plot(NL, NT_pred_log, color='orange', label='Logarithmic Fit')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Model Logaritmik')
plt.legend()

plt.tight_layout()
plt.show()

# Menghitung galat RMS untuk masing-masing metode
rms_linear = np.sqrt(mean_squared_error(NT, NT_pred_linear))
rms_power = np.sqrt(mean_squared_error(NT, NT_pred_power))
rms_log = np.sqrt(mean_squared_error(NT, NT_pred_log))

print(f'RMS Error (Linear): {rms_linear}')
print(f'RMS Error (Pangkat): {rms_power}')
print(f'RMS Error (Logarithmic): {rms_log}')