import matplotlib.pyplot as plt
import pandas as pd

# Load data using pandas for all three datasets
df = pd.read_csv('data.csv')
df_x1 = pd.read_csv('data_x1.csv')
df_x2 = pd.read_csv('data_x2.csv')

# Extract data directly without manual calculation for all datasets
p_values = df['p']
e_means = df['e_mean']
e_lowers = df['e_lower']
e_uppers = df['e_upper']
l_means = df['l_mean']
l_lowers = df['l_lower']
l_uppers = df['l_upper']

# For data_x1.csv
x1_l_means = df_x1['l_mean']
x1_l_lowers = df_x1['l_lower']
x1_l_uppers = df_x1['l_upper']
x1_e_means = df_x1['e_mean']
x1_e_lowers = df_x1['e_lower']
x1_e_uppers = df_x1['e_upper']

# For data_x2.csv
x2_l_means = df_x2['l_mean']
x2_l_lowers = df_x2['l_lower']
x2_l_uppers = df_x2['l_upper']
x2_e_means = df_x2['e_mean']
x2_e_lowers = df_x2['e_lower']
x2_e_uppers = df_x2['e_upper']

# Prepare error bars without manual calculation for all datasets
e_errors = [e_means - e_lowers, e_uppers - e_means]
l_errors = [l_means - l_lowers, l_uppers - l_means]

x1_e_errors = [x1_e_means - x1_e_lowers, x1_e_uppers - x1_e_means]
x1_l_errors = [x1_l_means - x1_l_lowers, x1_l_uppers - x1_l_means]

x2_e_errors = [x2_e_means - x2_e_lowers, x2_e_uppers - x2_e_means]
x2_l_errors = [x2_l_means - x2_l_lowers, x2_l_uppers - x2_l_means]

# Plotting
plt.figure(figsize=(15, 8))
plt.errorbar(p_values, e_means, yerr=e_errors, fmt='-o', label='Ensemble Class Models', capsize=5)
plt.errorbar(p_values, l_means, yerr=l_errors, fmt='-s', label='Joint Logits Class Models', capsize=5)

plt.errorbar(p_values, x1_e_means, yerr=x1_e_errors, fmt='-^', label='X1 Ensemble Class Models', capsize=5)
plt.errorbar(p_values, x1_l_means, yerr=x1_l_errors, fmt='-<', label='X1 Joint Logits Class Models', capsize=5)

plt.errorbar(p_values, x2_e_means, yerr=x2_e_errors, fmt='->', label='X2 Ensemble Class Models', capsize=5)
plt.errorbar(p_values, x2_l_means, yerr=x2_l_errors, fmt='-D', label='X2 Joint Logits Class Models', capsize=5)

plt.title('Combined Test Accuracy vs Label Noise')
plt.xlabel('Probability')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("noise_experiment_combined.png")
