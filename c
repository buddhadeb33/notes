import os
import pickle
import pandas as pd

def extract_features(model_obj):
    try:
        # Case 1: Dict-based model with feature info
        if isinstance(model_obj, dict):
            return model_obj.get("feature_input") or model_obj.get("features") or "Not found"

        # Case 2: LightGBM
        if model_obj.__class__.__name__.lower() in ["lgbmclassifier", "lgbmregressor"]:
            return model_obj.feature_name_

        # Case 3: Scikit-learn Logistic Regression
        if model_obj.__class__.__name__.lower() == "logisticregression":
            if hasattr(model_obj, "feature_names_in_"):
                return list(model_obj.feature_names_in_)
            return "feature_names_in_ not found"

        # Case 4: Try common attribute
        if hasattr(model_obj, "feature_input"):
            return model_obj.feature_input

    except Exception as e:
        return f"⚠️ Error: {e}"
    return "Not found"

# Define root directory to search
root_dir = "C:/Users/bmond01/Buddha_Projects/ccb_model/aim/models"
model_records = []

# Walk through all subfolders
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == "trained.pkl":
            model_path = os.path.join(dirpath, filename)
            try:
                # Extract model name from path after "HAS/"
                parts = model_path.replace("\\", "/").split("/")
                model_name = parts[parts.index("HAS") + 1] if "HAS" in parts else "unknown"

                # Load model
                with open(model_path, "rb") as f:
                    model_obj = pickle.load(f)

                model_type = model_obj.__class__.__name__.lower() if hasattr(model_obj, "__class__") else "unknown"
                feature_input = extract_features(model_obj)

                model_records.append({
                    "model_name": model_name,
                    "model_type": model_type,
                    "feature_input": feature_input
                })

            except Exception as e:
                print(f"❌ Error loading {model_path}: {e}")

# Convert to DataFrame
df = pd.DataFrame(model_records)
print(df)
# df.to_csv("model_summary.csv", index=False)
