# Load the actual model pickle file
with open("/mnt/data/file-JzWFkafj8Gq3BZPpXta4So", "rb") as f:
    model = pickle.load(f)

# Load the encoder pickle file correctly
with open("/mnt/data/file-9zLqHX8PoZK3KVWDQscsYe", "rb") as f:
    encoder = pickle.load(f)

# Encode the categorical column
encoded_bod = encoder.transform(synthetic_data[['primary_level_1_bod']])
encoded_bod_df = pd.DataFrame(encoded_bod.toarray(), columns=encoder.get_feature_names_out(['primary_level_1_bod']))

# Drop original categorical column and merge encoded version
synthetic_data_encoded = synthetic_data.drop(columns=['primary_level_1_bod']).reset_index(drop=True)
synthetic_data_encoded = pd.concat([synthetic_data_encoded, encoded_bod_df], axis=1)

# Predict using the loaded model
predictions = model.predict(synthetic_data_encoded)
predictions[:10]  # Show first 10 predictions as a sample
