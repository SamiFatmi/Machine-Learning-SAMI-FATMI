import pandas as pd
import joblib  as jl

# 1. Read in the tests samples
df_test = pd.read_csv("test_samples.csv", index_col = 0)

# 2. Read in the trained model
classifier = jl.load("trained_model.pkl")

