import numpy as np
import cv2
import matplotlib.pyplot as plt

x_values = [5, 10, 15, 20, 25]
y_values_golf = [0.565, 0.587, 0.592, 0.603, 0.600]
y_values_golf2 = [0.55, 0.566, 0.57, 0.585, 0.581]
y_values_golf3 =  [0.537, 0.575, 0.574, 0.586, 0.583]
y_values_golf4 =   [0.592, 0.63, 0.639, 0.661, 0.656]

y_values_highway = [0.583, 0.624, 0.613, 0.623, 0.611]
y_values_highway2 = [0.515, 0.563, 0.559, 0.565, 0.554]
y_values_highway3 =  [0.551, 0.595, 0.592, 0.593, 0.585]
y_values_highway4 =  [0.639, 0.672, 0.661, 0.663, 0.658]

x_name = "Number of Images"
y_name = "Avg AUC Score Golf Fields"
y_name = "Avg AUC Score Highway"

ideal_threshold = 1.0
good_threshold = 0.85
bad_threshold = 0.5

# plt.plot(x_values, y_values_golf, label="Golf Field")
plt.plot(x_values, y_values_golf2, label="Golf Field")
# plt.plot(x_values, y_values_highway, label="Highway")
plt.plot(x_values, y_values_highway2, label="Highway")

plt.axhline(y=ideal_threshold, color='g', linestyle='--', label="Ideal Threshold")
plt.axhline(y=good_threshold, color='b', linestyle='--', label="Good Threshold")
plt.axhline(y=bad_threshold, color='r', linestyle='--', label="Bad Threshold")

plt.xlabel(x_name)
plt.ylabel(y_name)
plt.title("Average AUC Score vs. Number of Images, Changed parameters")
plt.grid()
plt.legend()
plt.savefig("docs/obrazky-figures/AUC_Score_vs_Num_Images2.png")

plt.show()