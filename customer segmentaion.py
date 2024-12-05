# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Step 2: Load the Dataset
# You can upload the dataset to Google Colab or provide the path
from google.colab import files
uploaded = files.upload()  # Upload the CSV file

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')  # Ensure your file is named 'Mall_Customers.csv'
print(data.head())

# Step 3: Preprocess the Data
# Select relevant features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 4: K-Means Clustering
# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit K-Means with the chosen number of clusters (e.g., k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(features_scaled)

# Step 5: Gaussian Mixture Models (GMM) Clustering
gmm = GaussianMixture(n_components=5, random_state=42)
data['GMM_Cluster'] = gmm.fit_predict(features_scaled)

# Step 6: Visualize the Results
# Reduce dimensions for visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(12, 6))

# K-Means Clustering Visualization
plt.subplot(1, 2, 1)
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=data['KMeans_Cluster'], cmap='viridis', s=50)
plt.title('K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# GMM Clustering Visualization
plt.subplot(1, 2, 2)
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=data['GMM_Cluster'], cmap='plasma', s=50)
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()
