from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd


data1 = pd.read_csv('RF_scenario_s1_d1d1.csv', sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
data2 = pd.read_csv('RF_scenario_s1_d2d2.csv', sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
data3 = pd.read_csv('RF_scenario_s1_d3d3.csv', sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
data4 = pd.read_csv('RF_scenario_s3.csv', sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
data5 = pd.read_csv('RF_scenario_s2_d1d2.csv', sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
data6 = pd.read_csv('RF_scenario_s2_d1d3.csv', sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
data7 = pd.read_csv('RF_scenario_s2_d2d3.csv', sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')
data8 = pd.read_csv('RF_scenario_s2_d3d1.csv', sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

#print(list(data.columns))
#exit()
# Extract x and y coordinates
x1 = data1['x']  # Replace with your column name
#print(x)
y1 = data1['y']  # Replace with your column name
x2 = data2['x']
y2 = data2['y']
x3 = data3['x']
y3 = data3['y']
x4 = data4['x']
y4 = data4['y']
x5 = data5['x']
y5 = data5['y']
x6 = data6['x']
y6 = data6['y']
x7 = data7['x']
y7 = data7['y']
x8 = data8['x']
y8 = data8['y']


#x1 = [0.00, 0.004, 0.013, 0.020, 0.028, 0.035, 0.410, 0.561, 0.996]
#y1 = [0.00, 0.208, 0.400, 0.598, 0.802, 0.980, 0.980, 0.980, 0.980]


# Create a Matplotlib plot
plt.figure(figsize=(6, 6))  # Optional: Adjust the figure size

plt.xticks([0.0, 0.5, 1.0])
plt.yticks([0.0, 0.5, 1.0])
plt.plot(x1, y1, label='S1_d1d1 AUC=0.99')
plt.plot(x2, y2, label='S1_d2d2 AUC=0.97')
plt.plot(x3, y3, label='S1_d3d3 AUC=1.00')
plt.plot(x5, y5, label='S2_d1d2 AUC=0.97')
plt.plot(x6, y6, label='S2_d1d3 AUC=0.98')
plt.plot(x7, y7, label='S2_d2d3 AUC=0.92')
plt.plot(x8, y8, label='S2_d3d1 AUC=0.98')
plt.plot(x4, y4, label='S3 AUC=0.97')


# plt.plot(x1, y1, marker='o', linestyle='-', color='b', label='Data Points')
# plt.plot(x2, y2, marker='*', linestyle='-', color='b', label='Data Points')
# plt.plot(x3, y3, marker='o', linestyle='-', color='r', label='Data Points')
# plt.plot(x4, y4, marker='*', linestyle='-', color='r', label='Data Points')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend()
#plt.grid(True)

plt.show()