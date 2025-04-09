# Set seed for reproducibility
np.random.seed(42)

# Define structure for 100 synthetic rows
data = {
    'feature_has_covid_loans': np.random.randint(0, 2, 100),
    'ag_cre_or_ci_AG': np.random.randint(0, 2, 100),
    'ag_cre_or_ci_CRE': np.random.randint(0, 2, 100),

    'primary_level_1_bod_CANADIAN BUSINESS BANKING': np.random.randint(0, 2, 100),
    'primary_level_1_bod_WEALTH MANAGEMENT': np.random.randint(0, 2, 100),
    'primary_level_1_bod_BMO CAPITAL MARKETS': np.random.randint(0, 2, 100),

    'risk_rating_model_LC': np.random.randint(0, 3, 100),  # Assuming 0,1,2 as in your example

    'AVG_UTILIZATION_excl_balances_3m': np.round(np.random.uniform(0, 1, 100), 2),
    'CURRENT_BALANCE_AND_LIMIT_OVER_6m': np.random.randint(5000, 30000, 100),
    'SUM_n_trans_opicl_6m': np.random.randint(0, 50, 100),
    'months_since_acct_opened': np.random.randint(1, 240, 100),
    'TREND_CURRENT_TO_6m_utilization_only_if_negative': np.round(np.random.uniform(-1.0, 0.0, 100), 2),
    'excess_days': np.random.randint(0, 30, 100),
    'SUM_n_nsf_trans_6m': np.random.randint(0, 5, 100),
    'utilization_cc_only': np.round(np.random.uniform(0, 1, 100), 2)
}

# Create the DataFrame
synthetic_df = pd.DataFrame(data)

# Preview
print(synthetic_df.head())
