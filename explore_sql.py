import sqlite3
import pandas as pd

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

conn = sqlite3.connect('data/ecommerce.db')

conn = sqlite3.connect('data/ecommerce.db')

query = '''
SELECT Month,
       COUNT(*) AS sessions,
       ROUND(100.0 * SUM(CASE WHEN Revenue = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
FROM sessions
GROUP BY Month
ORDER BY conversion_rate_pct DESC;
'''

result = pd.read_sql_query(query, conn)
print(result)

conn.close()