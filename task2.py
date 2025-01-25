import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
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

# Print first few rows of products to debug
print("Products DataFrame:")
print(products.head())
print(products.columns)

# Merge Data
merged_data = transactions.merge(customers, on='customerid').merge(products, on='productid')

# Rename columns to avoid conflicts
merged_data.rename(columns={'price_x': 'transaction_price', 'price_y': 'product_price'}, inplace=True)

# Print columns and first few rows of merged_data to debug
print("Merged DataFrame:")
print(merged_data.head())
print(merged_data.columns)

# Create User Profiles (aggregate transaction data)
user_profiles = merged_data.groupby('customerid').agg({
    'totalvalue': 'sum',
    'quantity': 'sum',
    'product_price': 'mean'
}).reset_index()

# Normalize Data
scaler = StandardScaler()
user_profiles_scaled = scaler.fit_transform(user_profiles.iloc[:, 1:])

# Compute Similarities
similarities = cosine_similarity(user_profiles_scaled)
similarity_df = pd.DataFrame(similarities, index=user_profiles['customerid'], columns=user_profiles['customerid'])

# Generate Lookalike Recommendations
lookalike_results = {}
for customer_id in user_profiles['customerid'][:20]:
    similar_customers = similarity_df[customer_id].sort_values(ascending=False)[1:4]  # Top 3 excluding self
    lookalike_results[customer_id] = list(zip(similar_customers.index, similar_customers.values))

# Create Lookalike DataFrame
lookalike_df = pd.DataFrame({
    'customerid': lookalike_results.keys(),
    'lookalikes': lookalike_results.values()
})

# Save Results
lookalike_df.to_csv('Lookalike.csv', index=False)

print("Lookalike recommendations saved to 'Lookalike.csv'.")