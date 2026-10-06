import os
import zipfile

def extract_data():
    zip_path = os.path.abspath("data/bottle.zip")
    base_data_dir = os.path.abspath("data")
    
    if not os.path.exists(zip_path):
        print(f"❌ Could not find bottle.zip at {zip_path}")
        return
        
    print("📦 Found bottle.zip! Unpacking image matrices...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_data_dir)
    print("✅ Unpacking complete!")

if __name__ == "__main__":
    extract_data()
