# Encode the categorical column
encoded_bod = encoder.transform(synthetic_data[['primary_level_1_bod']])
encoded_bod_df = pd.DataFrame(encoded_bod, columns=encoder.get_feature_names_out(['primary_level_1_bod']))

# Drop original categorical column and merge encoded version
synthetic_data_encoded = synthetic_data.drop(columns=['primary_level_1_bod']).reset_index(drop=True)
synthetic_data_encoded = pd.concat([synthetic_data_encoded, encoded_bod_df], axis=1)

# Ensure model is not accidentally overwritten and is still the trained object
# If not, reload it using:
# import pickle
# with open("path_to_model.pkl", "rb") as f:
#     model = pickle.load(f)

# Make predictions
predictions = model.predict(synthetic_data_encoded)

# Show first 10 predictions
predictions[:10]
