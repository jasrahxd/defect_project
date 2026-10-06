import os
import ssl
import urllib.request
import zipfile

ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # Direct high-speed alternative zip mirror for MVTec Bottle Dataset
    url = "https://githubusercontent.com"
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
    zip_path = os.path.join(base_dir, "bottle.zip")
    
    os.makedirs(base_dir, exist_ok=True)
    
    print("⏳ Streaming factory zip archive directly into your workspace...")
    try:
        # Stream the standard ZIP archive
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download complete! Decompressing factory image folders...")
        
        # Extract native zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
            
        os.remove(zip_path) # Clean up file artifact
        print(f"🚀 Success! Industrial dataset unpacked perfectly inside: {base_dir}")
    except Exception as e:
        # Failover fallback using standard curl command line trick if urllib fails
        print("🔄 Trying alternative system download command...")
        try:
            os.system(f'curl -L -o "{zip_path}" "{url}"')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            os.remove(zip_path)
            print(f"🚀 Success! Industrial dataset unpacked perfectly inside: {base_dir}")
        except Exception as system_err:
            print(f"🛑 Error processing dataset: {system_err}")

if __name__ == "__main__":
    main()
