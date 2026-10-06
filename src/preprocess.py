import os
import cv2
import numpy as np

def load_images_from_folder(folder_path, image_size=(128, 128)):
    images = []
    if not os.path.exists(folder_path):
        print(f"Warning: Path {folder_path} does not exist.")
        return np.array(images)
        
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            # Read in grayscale to keep model training extremely fast
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, image_size)
                images.append(img_resized)
                
    # Normalize pixel values between 0.0 and 1.0 and expand dimensions for channels
    return np.array(images, dtype=np.float32) / 255.0

def get_dataset(base_data_path):
    train_good_path = os.path.join(base_data_path, 'train', 'good')
    test_good_path = os.path.join(base_data_path, 'test', 'good')
    test_defect_path = os.path.join(base_data_path, 'test', 'defective')
    
    x_train = load_images_from_folder(train_good_path)
    x_test_good = load_images_from_folder(test_good_path)
    x_test_defect = load_images_from_folder(test_defect_path)
    
    # Reshape arrays to add the channel dimension: (Num_Images, 128, 128, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test_good = np.expand_dims(x_test_good, axis=-1)
    x_test_defect = np.expand_dims(x_test_defect, axis=-1)
    
    return x_train, x_test_good, x_test_defect
