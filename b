import pandas as pd
import numpy as np

# Define possible categorical values based on encoded features
primary_residences = ['CA', 'CH', 'GB', 'US']
primary_levels = [
    'BMO CAPITAL MARKETS', 'CANADIAN BUSINESS BANKING', 'CANADIAN COMMERCIAL BANKING',
    'P&C US BUSINESS BANKING', 'P&C US COMMERCIAL', 'P&C US OTHER', 'WEALTH MANAGEMENT'
]
risk_rating_models = [
    'AG', 'AVERMEDIA', 'BBS', 'BD', 'EOI', 'EQS', 'FL', 'GC', 'GEF', 'HGS', 'HEALTHUS', 
    'HPB', 'IO', 'LC', 'MI-A', 'Model1-CORP', 'NoModel', 'NoModel/NoFS-CORP', 
    'ProxyRRM-CORP', 'ProxyRRM-BANK', 'Retail', 'RELIGIOU', 'SIO'
]

# Generate synthetic data
np.random.seed(42)  # For reproducibility
synthetic_data = pd.DataFrame({
    'primary_residence': np.random.choice(primary_residences, 100),
    'primary_level_1_bod': np.random.choice(primary_levels, 100),
    'risk_rating_model': np.random.choice(risk_rating_models, 100),
    'ag_cre_or_ci': np.random.randint(0, 2, 100),  # Binary feature
    'rating_sortable': np.random.randint(1, 11, 100)  # Assuming rating from 1 to 10
})

print(synthetic_data.head())
