import os
import cv2
import numpy as np

def create_synthetic_bottle(has_defect=False):
    img = np.ones((256, 256), dtype=np.uint8) * 50
    cv2.ellipse(img, (128, 140), (40, 70), 0, 0, 360, 200, -1)
    cv2.rectangle(img, (108, 40), (148, 70), 200, -1)
    cv2.rectangle(img, (103, 25), (153, 40), 120, -1)
    
    if has_defect:
        cv2.line(img, (110, 120), (145, 150), 20, 3)
        cv2.line(img, (145, 150), (135, 170), 20, 3)
        cv2.circle(img, (120, 110), 4, 10, -1)
        
    noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    return img

def main():
    # Force the path directly to your exact Windows project directory structure
    base_dir = r"C:\Users\jasrah\defect_project\data"
    
    train_good = os.path.join(base_dir, 'train', 'good')
    test_good = os.path.join(base_dir, 'test', 'good')
    test_defect = os.path.join(base_dir, 'test', 'defective')
    
    os.makedirs(train_good, exist_ok=True)
    os.makedirs(test_good, exist_ok=True)
    os.makedirs(test_defect, exist_ok=True)
    
    print("🏭 Generating factory line image vectors locally...")
    
    for i in range(100):
        img = create_synthetic_bottle(has_defect=False)
        cv2.imwrite(os.path.join(train_good, f"bottle_good_{i:03d}.png"), img)
        
    for i in range(15):
        img = create_synthetic_bottle(has_defect=False)
        cv2.imwrite(os.path.join(test_good, f"bottle_test_good_{i:03d}.png"), img)
        
    for i in range(15):
        img = create_synthetic_bottle(has_defect=True)
        cv2.imwrite(os.path.join(test_defect, f"bottle_test_fail_{i:03d}.png"), img)
        
    print(f"🚀 Success! 130 structural factory images created inside: {base_dir}")

if __name__ == "__main__":
    main()
