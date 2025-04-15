import os
import pickle
import pandas as pd

def extract_features(model_obj, model_path):
    try:
        # If the model is stored as a dictionary
        if isinstance(model_obj, dict):
            print("Model is a dictionary.")
            features = model_obj.get("feature_input") or model_obj.get("features")
            if features:
                return features
            else:
                print(f"No 'feature_input' or 'features' key in dict for {model_path}")
                return "Not found"

        model_type = model_obj.__class__.__name__.lower()
        print(f"Detected model class: {model_type}")

        # LightGBM
        if model_type in ["lgbmclassifier", "lgbmregressor"]:
            if hasattr(model_obj, "feature_name_"):
                return model_obj.feature_name_
            else:
                print(f"LightGBM model missing 'feature_name_' in {model_path}")
                return "feature_name_ not found"

        # Scikit-learn models with feature_names_in_
        if hasattr(model_obj, "feature_names_in_"):
            return list(model_obj.feature_names_in_)

        # Custom attribute
        if hasattr(model_obj, "feature_input"):
            return model_obj.feature_input

        print(f"Feature attributes not found for model at {model_path}")
        return "Not found"

    except Exception as e:
        print(f"Error extracting features from {model_path}: {e}")
        return f"Error: {e}"

# Set root directory
root_dir = "C:/Users/bmond01/Buddha_Projects/ccb_model/aim/models"
model_records = []

print(f"Walking through root directory: {root_dir}")

# Walk through directories
found_any_model = False
for dirpath, dirnames, filenames in os.walk(root_dir):
    print(f"Checking directory: {dirpath}")
    if "trained.pkl" in filenames:
        model_path = os.path.join(dirpath, "trained.pkl")
        print(f"Found model file: {model_path}")
        found_any_model = True

        try:
            with open(model_path, "rb") as f:
                model_obj = pickle.load(f)
                print("Model loaded successfully.")

            # Extract model name
            parts = model_path.replace("\\", "/").split("/")
            if "HAS" in parts:
                model_index = parts.index("HAS") + 1
                model_name = parts[model_index]
                print(f"Extracted model name: {model_name}")
            else:
                print(f"'HAS' not found in path, using fallback for model name.")
                model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))

            model_type = model_obj.__class__.__name__.lower() if hasattr(model_obj, "__class__") else "unknown"
            feature_input = extract_features(model_obj, model_path)

            model_records.append({
                "model_name": model_name,
                "model_type": model_type,
                "feature_input": feature_input
            })

        except Exception as e:
            print(f"Failed to load model at {model_path}: {e}")
    else:
        print("No 'trained.pkl' found in this directory.")

if not found_any_model:
    print("No trained.pkl model files found at all.")

# Create DataFrame
df = pd.DataFrame(model_records)
print("\nFinal Extracted Model Info:")
print(df)

# Save to CSV if needed
# df.to_csv("model_info_debug.csv", index=False)
