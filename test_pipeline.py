import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import DataLoader
from feature_engineer import FeatureEngineer

def test_data_loader_returns_correct_shape():
    loader = DataLoader('data/ecommerce.db')
    df = loader.load_table('sessions')
    assert df.shape[0] == 12330
    assert df.shape[1] == 18

def test_feature_engineer_adds_columns():
    loader = DataLoader('data/ecommerce.db')
    df = loader.load_table('sessions')
    fe = FeatureEngineer(df)
    result = fe.add_total_pages().add_total_duration().add_product_focus_ratio().get_dataframe()
    assert 'TotalPages' in result.columns
    assert 'TotalDuration' in result.columns
    assert 'ProductFocusRatio' in result.columns

def test_product_focus_ratio_is_between_0_and_1():
    loader = DataLoader('data/ecommerce.db')
    df = loader.load_table('sessions')
    fe = FeatureEngineer(df)
    result = fe.add_total_pages().add_total_duration().add_product_focus_ratio().get_dataframe()
    assert result['ProductFocusRatio'].min() >= 0
    assert result['ProductFocusRatio'].max() <= 1