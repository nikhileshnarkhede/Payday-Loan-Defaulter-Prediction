import streamlit as st
import pandas as pd
import joblib
from mypipes import *

# Load model
model = joblib.load('tuned_lgbm_model.pkl')

train_data = pd.read_csv("paydayloan_collections.csv")

# Define preprocessing pipeline (same as training)
p1 = pdPipeline([
    ('columns_selection', VarSelector(['var1','var2','var9','var10','var11',
                                       'var13','var17','var19','var23','var29'])),
    ('data_impute', DataFrameImputer()),
    ('get_dummy', get_dummies_Pipe())
])

# Fit pipeline on training data
temp = pd.DataFrame(data=p1.fit_transform(train_data),
                    columns=p1.get_feature_names())

# Create X_train structure for alignment
train_data_full = pd.concat([train_data, temp], axis=1)
X_train = train_data_full.drop(columns=['payment',
                                        'var1','var2','var9','var10','var11',
                                        'var13','var17','var19','var23','var29'])

# -----------------------
# Streamlit UI
# -----------------------
st.title("Payday Loan Payment Prediction App")
st.write("Enter values for var1 to var30. Categorical vars have radio buttons; numeric vars have sliders.")

user_input = {}

# Loop over var1–var30 and create UI based on dtype in CSV
for i in range(1, 31):
    col_name = f"var{i}"
    if col_name in train_data.columns:
        if train_data[col_name].dtype == 'object':
            # Get unique categories from training data
            categories = sorted(train_data[col_name].dropna().unique())
            if len(categories) > 0:
                user_input[col_name] = st.radio(f"{col_name}", categories)
            else:
                user_input[col_name] = st.text_input(f"{col_name}")
        else:
            # Numeric input with slider
            min_val = float(train_data[col_name].min())
            max_val = float(train_data[col_name].max())
            default_val = float(train_data[col_name].median())
            user_input[col_name] = st.slider(f"{col_name}", min_val, max_val, default_val)

# -----------------------
# Prediction
# -----------------------
if st.button("Predict Payment Outcome"):
    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # Apply preprocessing
    temp_input = pd.DataFrame(data=p1.transform(input_df),
                              columns=p1.get_feature_names())
    input_processed = pd.concat([input_df, temp_input], axis=1)

    # Drop original vars used in pipeline
    input_processed.drop(columns=['var1','var2','var9','var10','var11',
                                  'var13','var17','var19','var23','var29'], inplace=True)

    # Align to training feature structure
    input_processed = input_processed.reindex(columns=X_train.columns, fill_value=0)

    # Convert any remaining object columns to numeric
    for col in input_processed.columns:
        if input_processed[col].dtype == 'object':
            input_processed[col] = pd.to_numeric(input_processed[col], errors='coerce').fillna(0)

    # Predict
    pred_prob = model.predict_proba(input_processed)[0][1]
    pred_class = "Success" if pred_prob >= 0.5 else "Failure"

    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Probability of Success:** {pred_prob:.2%}")
    