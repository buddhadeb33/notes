import os
import pickle
import pandas as pd

# Root folder where all model folders reside
root_dir = "C:/Users/bmond01/Buddha_Projects/ccb_model/aim/models"

model_records = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    if "trained_model.pkl" in filenames:
        model_path = os.path.join(dirpath, "trained_model.pkl")

        try:
            # Extract model_name from path after "HAS/"
            parts = model_path.replace("\\", "/").split("/")
            model_name = parts[parts.index("HAS") + 1]

            # Load the model object
            with open(model_path, "rb") as f:
                model_obj = pickle.load(f)

            # Try to determine the model_type (LightGBM, XGBoost, etc.)
            if hasattr(model_obj, "__class__"):
                model_type = model_obj.__class__.__name__.lower()
            else:
                model_type = "unknown"

            # Try extracting features
            if isinstance(model_obj, dict):
                feature_input = model_obj.get("feature_input") or model_obj.get("features") or "Not found"
            elif hasattr(model_obj, "feature_name_"):
                feature_input = model_obj.feature_name_
            elif hasattr(model_obj, "feature_input"):
                feature_input = model_obj.feature_input
            else:
                feature_input = "Not found"

            model_records.append({
                "model_name": model_name,
                "model_type": model_type,
                "feature_input": feature_input
            })

        except Exception as e:
            print(f"⚠️ Error reading {model_path}: {e}")

# Create dataframe
df = pd.DataFrame(model_records)
print(df)
# df.to_csv("model_summary.csv", index=False)
