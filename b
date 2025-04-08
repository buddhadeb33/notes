import pandas as pd

input_data = pd.DataFrame([{
    'feature_has_covid_loans': 1,
    'ag_cre_or_ci_AG': 0,
    'ag_cre_or_ci_CRE': 1,
    'primary_level_1_bod_CANADIAN BUSINESS BANKING': 0,
    'primary_level_1_bod_WEALTH MANAGEMENT': 1,
    'primary_level_1_bod_BMO CAPITAL MARKETS': 0,
    'risk_rating_model_LC': 2,
    'AVG_UTILIZATION_excl_balances_3m': 0.5,
    'CURRENT_BALANCE_AND_LIMIT_OVER_6m': 15000,
    'SUM_n_trans_opic1_6m': 20,
    'months_since_acct_opened': 12,
    'TREND_CURRENT_TO_6m_utilization_only_if_negative': -0.1,
    'excess_days': 5,
    'SUM_n_nsf_trans1_6m': 1,
    'utilization_cc_only': 0.8
}])


X_transformed = encoder.transform(data)
prediction = model.predict(X_transformed)
proba = model.predict_proba(X_transformed)

print(f"🎯 Prediction: {prediction[0]}")
print(f"Confidence Scores: {proba}")



# Get feature names after encoding
try:
    encoded_features = encoder.get_feature_names_out()
except:
    encoded_features = model.feature_names_in_

coeffs = model.coef_[0]

importance = pd.DataFrame({
    'feature': encoded_features,
    'coefficient': coeffs
}).sort_values(by='coefficient', key=np.abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='coefficient', y='feature', data=importance.head(15))
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()


import shap

explainer = shap.Explainer(model, X_transformed)
shap_values = explainer(X_transformed)

shap.plots.waterfall(shap_values[0])
