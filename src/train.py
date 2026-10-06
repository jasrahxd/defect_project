import os
import sys
import pickle
import numpy as np
from preprocess import get_dataset
from model import build_anomaly_detector

# Force immediate text updates to your terminal window
sys.stdout.reconfigure(line_buffering=True)

def main():
    base_data_path = r"C:\Users\jasrah\defect_project\data"
    model_dir = r"C:\Users\jasrah\defect_project\models"
    model_save_path = os.path.join(model_dir, "defect_detector.pkl")
    
    print("--- 🏭 Starting Industrial AI Training Engine ---")
    
    x_train, _, _ = get_dataset(base_data_path)
    
    if len(x_train) == 0:
        print("❌ Error: Images missing from data directory! Run generate_data.py first.")
        return

    print(f"📦 Loaded {len(x_train)} flawless structural samples.")
    
    # Flatten images from matrices (128x128x1) to arrays (16384,) for Scikit-Learn processing
    num_samples = x_train.shape[0]
    x_train_flat = x_train.reshape(num_samples, -1)
    
    print("🧠 Optimizing Isolation Forest boundaries...")
    detector = build_anomaly_detector()
    detector.fit(x_train_flat)
    
    # Save the trained model parameters safely to disk
    os.makedirs(model_dir, exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(detector, f)
        
    print(f"🚀 Success! Model parameters serialized to: {model_save_path}")

if __name__ == '__main__':
    main()
