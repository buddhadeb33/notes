import numpy as np

# Define possible categories for the OneHotEncoded column based on the image
bod_categories = [
    'BMO CAPITAL MARKETS',
    'CANADIAN BUSINESS BANKING',
    'CANADIAN COMMERCIAL BANKING',
    'OTHER',
    'P&C US BUSINESS BANKING',
    'P&C US COMMERCIAL',
    'P&C US OTHER',
    'TECHNOLOGY AND OPERATIONS',
    'TOTAL CORPORATE',
    'WEALTH MANAGEMENT'
]

# Define other numerical features based on your original input data
def generate_synthetic_row():
    return {
        'feature_has_covid_loans': np.random.randint(0, 2),
        'ag_cre_or_ci_AG': np.random.randint(0, 2),
        'ag_cre_or_ci_CRE': np.random.randint(0, 2),
        'primary_level_1_bod': np.random.choice(bod_categories),
        'risk_rating_model_LC': np.random.randint(1, 6),
        'AVG_UTILIZATION_excl_balances_3m': np.round(np.random.uniform(0, 1), 2),
        'CURRENT_BALANCE_AND_LIMIT_OVER_6m': np.round(np.random.uniform(100, 1000), 2),
        'SUM_n_trans_optci_6m': np.round(np.random.uniform(0, 50), 1),
        'months_since_acct_opened': np.random.randint(1, 240),
        'TREND_CURRENT_TO_6m_utilization_only_if_negative': np.round(np.random.uniform(0, 1), 2),
        'excess_days': np.random.randint(0, 60),
        'SUM_n_nsf_trans1_6m': np.random.randint(0, 10)
    }

# Create 100 synthetic rows
synthetic_data = pd.DataFrame([generate_synthetic_row() for _ in range(100)])
synthetic_data.head()
