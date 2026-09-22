import pandas as pd
import sqlite3


class DataLoader:
    def __init__(self, db_path):
        self.db_path = db_path

    def load_table(self, table_name):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df