import os
import pickle
import pandas as pd
from sklearn.base import BaseEstimator

# Update this to your root directory
root_dir = r"C:\Users\bmond01\Buddha_Projects\ccb_model\aim\models"

model_info = []

print(f"Walking through root directory: {root_dir}")

for dirpath, dirnames, filenames in os.walk(root_dir):
    print(f"\nChecking directory: {dirpath}")
    
    for file in filenames:
        if file == "trained_model.pkl":
            model_path = os.path.join(dirpath, file)
            model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(dirpath))))  # e.g., us_zba

            print(f"Found model file: {model_path}")
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                model_type = type(model).__name__

                try:
                    features = model.feature_name_in_
                except AttributeError:
                    try:
                        features = model.booster_.feature_name()  # for LGBM
                    except Exception:
                        features = "Not available"

                model_info.append({
                    "model_name": model_name,
                    "model_type": model_type,
                    "feature_names": features
                })

                print(f"Extracted model_type: {model_type}")
                print(f"Feature names: {features if isinstance(features, list) else 'Unavailable'}")
            
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
        else:
            print("No 'trained_model.pkl' found in this directory.")

df = pd.DataFrame(model_info)

print("\nFinal Extracted Model Info:")
print(df.head())
