from sklearn.model_selection import train_test_split
import os
import shutil
from glob import glob

def split_dataset(original_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    classes = os.listdir(original_dir)
    
    for cls in classes:
        cls_dir = os.path.join(original_dir, cls)
        images = glob(os.path.join(cls_dir, '*'))
        trian_img, temp_imgs = train_test_split(images, test_size=(1- train_ratio), random_state=42)