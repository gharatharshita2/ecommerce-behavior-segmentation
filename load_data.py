import pandas as pd
import sqlite3

df = pd.read_csv('data/online_shoppers_intention.csv')

conn = sqlite3.connect('data/ecommerce.db')

df.to_sql('sessions', conn, if_exists='replace', index=False)

conn.close()
print('Done! Loaded', len(df), 'rows into ecommerce.db')