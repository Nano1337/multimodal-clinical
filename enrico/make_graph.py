import matplotlib.pyplot as plt
import pandas as pd

# Load data using pandas
df = pd.read_csv('data_x2.csv')

# Extract data directly without manual calculation
p_values = df['p']
e_means = df['e_mean']
e_lowers = df['e_lower']
e_uppers = df['e_upper']
l_means = df['l_mean']
l_lowers = df['l_lower']
l_uppers = df['l_upper']

# Prepare error bars without manual calculation
e_errors = [e_means - e_lowers, e_uppers - e_means]
l_errors = [l_means - l_lowers, l_uppers - l_means]

# Plotting
plt.figure(figsize=(10, 5))
plt.errorbar(p_values, e_means, yerr=e_errors, fmt='-o', label='Ensemble Class Models', capsize=5)
plt.errorbar(p_values, l_means, yerr=l_errors, fmt='-s', label='Joint Logits Class Models', capsize=5)

plt.title('X2 Test Accuracy vs Label Noise')
plt.xlabel('Probability')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("x2_noise_experiment.png")
