import pandas as pd 

# read pandas csv
df_ensemble = pd.read_csv('mimic_ensemble.csv')
df_logits = pd.read_csv('mimic_jlogits.csv')

ensemble_acc = df_ensemble['test_acc_epoch']
logits_acc = df_logits['test_acc_epoch']

count_greater = (logits_acc > ensemble_acc).sum()
print(f"Number of instances where logits_acc is greater than ensemble_acc: {count_greater}")

x1_acc_ensemble = df_ensemble['x1_test_acc']
x1_acc_logits = df_logits['x1_test_acc']

count_x1_greater = (x1_acc_logits > x1_acc_ensemble).sum()
print(f"Number of instances where x1_acc logits is greater than ensemble: {count_x1_greater}")

x2_acc_ensemble = df_ensemble['x2_test_acc']
x2_acc_logits = df_logits['x2_test_acc']

count_x2_greater = (x2_acc_logits > x2_acc_ensemble).sum()
print(f"Number of instances where x2_acc logits is greater than ensemble: {count_x1_greater}")

