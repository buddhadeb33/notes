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
