import pandas as pd
from src.common import normalize_column_name, feature_columns, TARGET, ID_COLUMN

def test_normalize_column_name():
    assert normalize_column_name("default payment next month") == "default_payment_next_month"

def test_feature_columns_excludes_target_and_id():
    df = pd.DataFrame({
        "id": [1, 2],
        "limit_bal": [10000, 20000],
        TARGET: [0, 1],
    })
    assert feature_columns(df) == ["limit_bal"]
