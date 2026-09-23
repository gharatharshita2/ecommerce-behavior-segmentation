import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

loader = DataLoader('data/ecommerce.db')
df = loader.load_table('sessions')

print("Summary statistics for key numeric columns:")
print(df[['ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']].describe())

plt.figure(figsize=(8, 5))
df['ProductRelated_Duration'].hist(bins=50)
plt.title('Distribution of Time Spent on Product Pages')
plt.xlabel('Seconds')
plt.ylabel('Number of Sessions')
plt.savefig('product_duration_distribution.png')
print('Saved plot: product_duration_distribution.png')

numeric_cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'Revenue']

correlations = df[numeric_cols].corr()['Revenue'].sort_values(ascending=False)
print()
print('Correlation of each feature with Revenue (purchase):')
print(correlations)

plt.figure(figsize=(10, 6))
import seaborn as sns
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print('Saved plot: correlation_heatmap.png')