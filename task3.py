import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
customers = pd.read_csv('data\Customers.csv')
transactions = pd.read_csv('data\Transactions.csv')

# Clean column names for consistency
customers.columns = customers.columns.str.strip().str.lower()
transactions.columns = transactions.columns.str.strip().str.lower()

# Merge Data
merged_data = transactions.merge(customers, on='customerid')

# Create Segmentation Data (aggregate customer transaction data)
segmentation_data = merged_data.groupby('customerid').agg({
    'totalvalue': 'sum',
    'quantity': 'sum'
}).reset_index()

# Include profile information (region encoding)
region_dummies = pd.get_dummies(customers[['customerid', 'region']], columns=['region'], drop_first=True)
segmentation_data = segmentation_data.merge(region_dummies, on='customerid')

# Normalize Data
scaler = StandardScaler()
segmentation_data_scaled = scaler.fit_transform(segmentation_data.iloc[:, 1:])

# Apply K-Means Clustering
n_clusters = 4  # You can adjust this based on your analysis
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(segmentation_data_scaled)
segmentation_data['cluster'] = clusters

# Evaluate Clustering
db_index = davies_bouldin_score(segmentation_data_scaled, clusters)
print(f'Davies-Bouldin Index: {db_index}')

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=segmentation_data, x='totalvalue', y='quantity', hue='cluster', palette='Set2')
plt.title('Customer Segments')
plt.xlabel('Total Value')
plt.ylabel('Quantity')
plt.legend(title='Cluster')
plt.show()

# Visualize Clusters for Regions
plt.figure(figsize=(10, 6))
sns.boxplot(data=segmentation_data, x='cluster', y='totalvalue', palette='Set2')
plt.title('Total Value Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Total Value')
plt.show()

# Save Clustering Results
segmentation_data.to_csv('Aniket_Vishwakarma_Clustering_Results.csv', index=False)

# Report Results
print(f'Number of Clusters: {n_clusters}')
print(f'Davies-Bouldin Index: {db_index}')
print("Clustering results saved to 'Clustering_Results.csv'.")
