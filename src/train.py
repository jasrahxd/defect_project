import os
import sys
import pickle
import numpy as np
from preprocess import get_dataset
from model import build_anomaly_detector

sys.stdout.reconfigure(line_buffering=True)

def main():
    # Dynamic pathing relative to this file location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_data_path = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models")
    model_save_path = os.path.join(model_dir, "defect_detector.pkl")
    
    print("--- 🏭 Starting Industrial AI Training Engine ---")
    
    x_train, _, _ = get_dataset(base_data_path)
    
    if len(x_train) == 0:
        print("❌ Error: Images missing from data directory!")
        return

    print(f"📦 Loaded {len(x_train)} flawless structural samples.")
    
    num_samples = x_train.shape[0]
    x_train_flat = x_train.reshape(num_samples, -1)
    
    detector = build_anomaly_detector()
    detector.fit(x_train_flat)
    
    os.makedirs(model_dir, exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(detector, f)
        
    print(f"🚀 Success! Model parameters serialized to: {model_save_path}")

if __name__ == '__main__':
    main()
