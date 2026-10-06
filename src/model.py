import numpy as np
from sklearn.ensemble import IsolationForest

def build_anomaly_detector():
    # An Isolation Forest isolates anomalies by randomly partitioning features
    # perfect for industrial structural inspection where defects vary wildly
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    return model
