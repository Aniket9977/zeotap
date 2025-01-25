import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore Data
customers = pd.read_csv('data/Customers.csv')
products = pd.read_csv('data/Products.csv')
transactions = pd.read_csv('data/Transactions.csv')

# Clean column names for consistency
customers.columns = customers.columns.str.strip().str.lower()
products.columns = products.columns.str.strip().str.lower()
transactions.columns = transactions.columns.str.strip().str.lower()

# Convert price column to numeric, forcing errors to NaN
products['price'] = pd.to_numeric(products['price'], errors='coerce')

# Drop rows with NaN values in the price column
products = products.dropna(subset=['price'])

# Print data types to debug


# EDA
print(customers.info())
print(products.info())
print(transactions.info())

# Merge Data for Analysis
merged_data = transactions.merge(customers, on='customerid').merge(products, on='productid')

# Visualizations
# Region-wise customer distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=customers, x='region')
plt.title('Customer Distribution by Region')
plt.show()

# Average product price by category
plt.figure(figsize=(10, 6))
numeric_columns = products.select_dtypes(include=[np.number]).columns.tolist()
sns.barplot(data=products.groupby('category')[numeric_columns].mean().reset_index(), x='category', y='price')
plt.title('Average Product Price by Category')
plt.show()
