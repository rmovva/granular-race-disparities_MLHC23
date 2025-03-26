import os

# This should point to MIMIC-IV-2.2, which should have the ed, hosp, and icu subfolders
mimic_iv_path = '/share/pierson/mimic_data/2.2/'

# This is where processed data, including the main dataframe csv, will be saved
data_path = '/share/pierson/mimic_data/processed/mimic4ed/'

# This is where results data will be saved, e.g. performance metrics of ML classifiers
results_path = os.path.join(data_path, 'test_code_results')