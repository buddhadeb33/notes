try:
    model = load_pickle(os.path.join('../models/deal/model_ca.pkl'))
    print("Type:", type(model))
    print("*" * 40)

    if hasattr(model, 'get_params'):
        print("Parameters:", model.get_params())
        print("*" * 40)

    if hasattr(model, 'feature_names_in_'):
        print("Feature names:", model.feature_names_in_)
        print("*" * 40)

    if hasattr(model, 'coef_'):
        print("Coefficients:", model.coef_)
        print("*" * 40)

    if hasattr(model, 'intercept_'):
        print("Intercept:", model.intercept_)
        print("*" * 40)

    if hasattr(model, 'feature_importances_'):
        print("Feature importances:", model.feature_importances_)
        print("*" * 40)

    if hasattr(model, 'classes_'):
        print("Target classes:", model.classes_)
        print("*" * 40)

    if hasattr(model, 'n_iter_'):
        print("Number of iterations:", model.n_iter_)
        print("*" * 40)

    if hasattr(model, 'predict_proba'):
        print("Supports predict_proba: ✅")
        print("*" * 40)

    if hasattr(model, 'decision_function'):
        print("Supports decision_function: ✅")
        print("*" * 40)

    if hasattr(model, 'steps'):
        print("Pipeline steps:")
        for step in model.steps:
            print(step)
        print("*" * 40)

    print("Available methods and attributes:")
    for fn in dir(model):
        if not fn.startswith("_"):
            print(fn)
    print("*" * 40)

    print("Custom attributes in model.__dict__:")
    for key, value in model.__dict__.items():
        print(f"{key}: {value}")
    print("*" * 40)

except Exception as e:
    print("Error loading deal_inference.pkl:", e)




with open('../models/deal/deal.json', 'r') as f:
    config = json.load(f)

print("Loaded JSON config:")
for key, value in config.items():
    print(f"{key}: {value}")
print("*" * 40)

with open('../models/deal/enc_ca.pkl', 'rb') as f:
    encoder = pickle.load(f)

print("Type of encoder:", type(encoder))
print("*" * 40)

if hasattr(encoder, 'get_params'):
    print("Parameters:", encoder.get_params())
    print("*" * 40)

if hasattr(encoder, 'feature_names_in_'):
    print("Feature names in:", encoder.feature_names_in_)
    print("*" * 40)

if hasattr(encoder, 'get_feature_names_out'):
    try:
        print("Feature names out:", encoder.get_feature_names_out())
    except Exception as e:
        print("Could not extract feature names out:", e)
    print("*" * 40)

if hasattr(encoder, 'categories_'):
    print("Categories (for OneHotEncoder):", encoder.categories_)
    print("*" * 40)

if hasattr(encoder, 'mean_'):
    print("Mean (for scaler):", encoder.mean_)
    print("*" * 40)

if hasattr(encoder, 'scale_'):
    print("Scale (for scaler):", encoder.scale_)
    print("*" * 40)

print("Available methods and attributes:")
for fn in dir(encoder):
    if not fn.startswith("_"):
        print(fn)
print("*" * 40)

print("Custom attributes in encoder.__dict__:")
for key, value in encoder.__dict__.items():
    print(f"{key}: {value}")
print("*" * 40)
