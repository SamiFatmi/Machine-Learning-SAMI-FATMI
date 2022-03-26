import pandas as pd
import joblib  as jl
from sklearn.ensemble import VotingClassifier

# 1. Read in the tests samples
test_df = pd.read_csv("Files/test_samples.csv", index_col = 0)

# 2. Read in the trained model
model = jl.load("Files/trained_model.pkl")

# 3. Splitting the data into X and y
X = test_df.drop(["id","cardio"], axis = 1)
y = test_df["cardio"]

# 4. Making predictions and prediction probabilities
y_pred = model.predict(X)
y_probabilities = model.predict_proba(X)

# 5. Creating the dataframe 
predictions_df = pd.DataFrame({
                                "Probability Class 0": y_probabilities[:,0],
                                "Probability Class 1": y_probabilities[:,1],
                                "Predicted": y_pred
                            })

# 6. Exporting the dataframe as csv
predictions_df.to_csv("Files/prediction.csv")