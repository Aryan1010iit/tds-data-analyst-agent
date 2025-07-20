# utils/analysis.py
import numpy as np
import pandas as pd

def compute_correlation(df: pd.DataFrame, x: str, y: str) -> float:
    """Compute Pearson correlation between two numeric columns."""
    return float(df[x].astype(float).corr(df[y].astype(float)))

def compute_regression(df: pd.DataFrame, x: str, y: str) -> tuple[float,float]:
    """
    Fit a line y = m*x + b.
    Returns (slope m, intercept b).
    """
    xs = df[x].astype(float)
    ys = df[y].astype(float)
    m, b = np.polyfit(xs, ys, 1)
    return float(m), float(b)
