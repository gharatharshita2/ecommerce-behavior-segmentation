import pandas as pd
from data_loader import DataLoader
from feature_engineer import FeatureEngineer

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# Load the data using our DataLoader class
loader = DataLoader('data/ecommerce.db')
df = loader.load_table('sessions')

print("Loaded shape:", df.shape)

# Use FeatureEngineer to add new columns, chaining the calls together
fe = FeatureEngineer(df)
df_enriched = (
    fe.add_total_pages()
      .add_total_duration()
      .add_product_focus_ratio()
      .get_dataframe()
)

print()
print("New columns added. First 5 rows of just the new ones:")
print(df_enriched[['TotalPages', 'TotalDuration', 'ProductFocusRatio', 'Revenue']].head())