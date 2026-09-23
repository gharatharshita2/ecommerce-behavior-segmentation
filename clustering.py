import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_loader import DataLoader
from feature_engineer import FeatureEngineer

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

loader = DataLoader('data/ecommerce.db')
df = loader.load_table('sessions')

fe = FeatureEngineer(df)
df = (fe.add_total_pages()
        .add_total_duration()
        .add_product_focus_ratio()
        .get_dataframe())

features = ['TotalPages', 'TotalDuration', 'ProductFocusRatio', 'BounceRates', 'ExitRates', 'PageValues']
X = df[features]

print('Features used for clustering:')
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print()
print('Scaled features (first 5 rows):')
print(pd.DataFrame(X_scaled, columns=features).head())

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.savefig('elbow_plot.png')
print('Saved plot: elbow_plot.png')

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print()
print('Cluster sizes:')
print(df['Cluster'].value_counts().sort_index())

print()
print('Average feature values per cluster:')
print(df.groupby('Cluster')[features].mean().round(2))

print()
print('Actual conversion rate per cluster:')
print(df.groupby('Cluster')['Revenue'].mean().round(3) * 100)

df.to_csv('sessions_with_clusters.csv', index=False)
print('Saved: sessions_with_clusters.csv')