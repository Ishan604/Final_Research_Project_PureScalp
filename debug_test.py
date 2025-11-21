"""
Debug script to test the dataset loading
"""
import os
import pandas as pd
from config import Config

def debug_dataset():
    print("🔍 Debug: Testing dataset loading...")
    
    # Check paths
    print(f"Dataset path: {Config.DATASET_PATH}")
    print(f"Dataset exists: {os.path.exists(Config.DATASET_PATH)}")
    
    print(f"Train CSV path: {Config.TRAIN_CSV_PATH}")
    print(f"Train CSV exists: {os.path.exists(Config.TRAIN_CSV_PATH)}")
    
    print(f"Valid CSV path: {Config.VALID_CSV_PATH}")
    print(f"Valid CSV exists: {os.path.exists(Config.VALID_CSV_PATH)}")
    
    print(f"Train images path: {Config.TRAIN_IMAGES_PATH}")
    print(f"Train images exists: {os.path.exists(Config.TRAIN_IMAGES_PATH)}")
    
    print(f"Valid images path: {Config.VALID_IMAGES_PATH}")
    print(f"Valid images exists: {os.path.exists(Config.VALID_IMAGES_PATH)}")
    
    # Try to read CSV files
    if os.path.exists(Config.TRAIN_CSV_PATH):
        try:
            train_df = pd.read_csv(Config.TRAIN_CSV_PATH)
            
            # Clean column names
            train_df.columns = train_df.columns.str.strip()
            
            print(f"✅ Train CSV loaded: {len(train_df)} rows")
            print("Train CSV columns (after cleaning):", train_df.columns.tolist())
            print("Train CSV first 3 rows:")
            print(train_df.head(3))
            
            # Test label processing
            def get_scalp_type(row):
                if row['d'] == 1 and row['ds'] == 0 and row['o'] == 0 and row['s'] == 0:
                    return 'dandruff'
                elif row['ds'] == 1:
                    return 'dandruff_sensitive'
                elif row['o'] == 1:
                    return 'oily'
                elif row['s'] == 1:
                    return 'sensitive'
                else:
                    return None
            
            train_df['scalp_type'] = train_df.apply(get_scalp_type, axis=1)
            train_df = train_df.dropna(subset=['scalp_type'])
            print(f"After processing labels: {len(train_df)} valid rows")
            print("Label distribution:")
            print(train_df['scalp_type'].value_counts())
            
        except Exception as e:
            print(f"❌ Error reading train CSV: {e}")
    
    if os.path.exists(Config.VALID_CSV_PATH):
        try:
            valid_df = pd.read_csv(Config.VALID_CSV_PATH)
            print(f"✅ Valid CSV loaded: {len(valid_df)} rows")
        except Exception as e:
            print(f"❌ Error reading valid CSV: {e}")
    
    # Check some image files
    if os.path.exists(Config.TRAIN_IMAGES_PATH):
        image_files = [f for f in os.listdir(Config.TRAIN_IMAGES_PATH) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} image files in train folder")
        if image_files:
            print("Sample image files:", image_files[:5])

if __name__ == "__main__":
    debug_dataset()