import pandas as pd

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('data/online_shoppers_intention.csv')

print("Shape (rows, columns):", df.shape)
print()
print("Column names and types:")
print(df.dtypes)
print()
print("First 5 rows:")
print(df.head())