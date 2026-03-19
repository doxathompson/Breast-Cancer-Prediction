"""
tests/test_pipeline.py
───────────────────────
Pytest configurations to validate core ML preprocessing limits.
"""

from src.data.loader import load_wbcd, split_data
from src.data.preprocessor import build_preprocessing_pipeline
import pytest
import numpy as np
import pandas as pd

def test_data_ingestion():
    df = load_wbcd("data/raw")
    assert df.shape == (569, 31)
    assert "diagnosis" in df.columns
    assert df["diagnosis"].nunique() == 2

def test_preprocessing_pipeline():
    df = load_wbcd("data/raw")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # Contains missing?
    df_missing = X_train.copy()
    df_missing.iloc[0, 5] = np.nan
    
    pipe = build_preprocessing_pipeline(correlation_threshold=0.95)
    X_pp = pipe.fit_transform(df_missing, y_train)
    
    assert not np.isnan(X_pp).any()
    assert X_pp.shape[1] < X_train.shape[1]  # Dropped colinearity features
