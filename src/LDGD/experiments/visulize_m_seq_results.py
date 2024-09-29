import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Define the path to your data
path = 'D:\\projects\\Neural_Signal_Classifier\\results\\m_sequence\\debug\\'

# List all folders that start with 'p'
folder_list = [f for f in os.listdir(path) if f.startswith('p')]
data_list = []

# Loop through the folders and extract the data
for folder in folder_list:
    try:
        path_folder = os.path.join(path, folder)
        data = pd.read_csv(os.path.join(path_folder, 'xgboost_fold_report_results.csv'))

        # Get the f1-score_mean values for the first 100 entries and reshape into a 10x10 matrix
        f1_mean_values_clean = data['f1-score_mean'].iloc[:100].values
        f1_mean_matrix_clean = f1_mean_values_clean.reshape(10, 10)
        data_list.append(f1_mean_matrix_clean)
    except Exception as e:
        print(f"Error processing folder {folder}: {e}")
        continue

# Stack all the data from different folders and calculate the mean and std
data_all = np.stack(data_list, axis=0)
data_mean = np.mean(data_all, axis=0)
data_std = np.std(data_all, axis=0)

# Plotting the heatmap for the mean f1-scores
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(data_mean, annot=True, cmap="RdYlGn", cbar=True)
plt.title("Heatmap of Mean F1-Score for 10x10 Pixels")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")
plt.savefig(os.path.join("C:\\Users\\Navid Ziaei\\OneDrive\\PHD\\Pajooheshi2\\Figures\\", 'mean_f1_score_heatmap.png'))
plt.show()

# Define the center of the matrix for distance calculations
center = (4, 5)

# Calculate the Euclidean distance from the center for each pixel
distances = []
for i in range(10):
    for j in range(10):
        distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
        distances.append(distance)

# Compute the average accuracy (f1-score_mean) and standard deviation for each pixel
accuracy_mean = data_mean.flatten()
accuracy_std = data_std.flatten()

# Plot accuracy (mean f1-score) with respect to distance from the center
plt.figure(figsize=(8, 6), dpi=300)
plt.errorbar(distances, accuracy_mean, yerr=accuracy_std, fmt='o', ecolor='red', capsize=3, label='Accuracy (F1-Score)')
plt.plot(np.unique(distances),
         [np.mean(accuracy_mean[np.array(distances) == d]) for d in np.unique(distances)],
         color='blue', label='Average Accuracy')

plt.xlabel('Distance from Center')
plt.ylabel('Accuracy (Mean F1-Score)')
plt.title('Accuracy with Respect to Distance from the Center of the Matrix')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("C:\\Users\\Navid Ziaei\\OneDrive\\PHD\\Pajooheshi2\\Figures\\", 'mean_f1_dist_plot.png'))

plt.show()
