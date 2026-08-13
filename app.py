import streamlit as st
import pandas as pd
import joblib
import shap
from mapie.classification import SplitConformalClassifier

# Set up the page
st.set_page_config(
    page_title="Alzheimer's Prediction Demo",
    page_icon="🧠",
    layout="centered"
)


# App title
st.title("Alzheimer's Prediction Demo")

st.write(
    "Enter patient information below to generate a prediction "
    "using the simplified Random Forest model."
)


# Load the trained model and calibration data
rf_demo = joblib.load("rf_demo_model.pkl")
X_calib_demo = joblib.load("X_calib_demo.pkl")
y_calib_demo = joblib.load("y_calib_demo.pkl")


# Create the conformal prediction model
mapie_demo = SplitConformalClassifier(
    estimator=rf_demo,
    confidence_level=0.90,
    conformity_score="lac",
    prefit=True
)

mapie_demo.conformalize(X_calib_demo, y_calib_demo)

explainer_demo = shap.TreeExplainer(rf_demo)


# Input section
st.header("Enter Patient Information")

mmse = st.slider(
    "MMSE Score (0–30)",
    min_value=0.0,
    max_value=30.0,
    value=20.0,
    step=0.1
)

functional = st.slider(
    "Functional Assessment Score (0–10)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

adl = st.slider(
    "Activities of Daily Living Score (0–10)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

memory = st.selectbox(
    "Memory Complaints",
    ["No", "Yes"]
)

behavioral = st.selectbox(
    "Behavioral Problems",
    ["No", "Yes"]
)


# Convert Yes/No to 0/1
memory_value = 1 if memory == "Yes" else 0
behavioral_value = 1 if behavioral == "Yes" else 0


# Create input data
patient_data = pd.DataFrame({
    "MMSE": [mmse],
    "FunctionalAssessment": [functional],
    "ADL": [adl],
    "MemoryComplaints": [memory_value],
    "BehavioralProblems": [behavioral_value]
})


# Run prediction
if st.button("Run Prediction"):

    import matplotlib.pyplot as plt

    # Random Forest prediction
    prediction = rf_demo.predict(patient_data)[0]

    # Prediction probabilities
    probabilities = rf_demo.predict_proba(patient_data)[0]

    # MAPIE prediction set
    _, prediction_set = mapie_demo.predict_set(patient_data)

    prediction_set = prediction_set[0, :, 0]


    # Show predicted class
    st.header("Predicted Class")

    if prediction == 1:
        st.error("Alzheimer's")
    else:
        st.success("No Alzheimer's")


    # Show prediction probabilities
    st.subheader("Prediction Probability")

    st.write(f"No Alzheimer's: **{probabilities[0]:.1%}**")
    st.write(f"Alzheimer's: **{probabilities[1]:.1%}**")


    # Show conformal prediction set
    st.subheader("Conformal Prediction Set (90%)")

    included_classes = []

    if prediction_set[0]:
        included_classes.append("No Alzheimer's")

    if prediction_set[1]:
        included_classes.append("Alzheimer's")


    if len(included_classes) == 0:
        st.warning("No class was included in the prediction set.")

    elif len(included_classes) == 1:
        st.info(included_classes[0])

    else:
        st.warning("No Alzheimer's OR Alzheimer's")

    
    st.subheader("Why did the model make this prediction?")

    st.write("The SHAP waterfall plot shows how each feature contributed to the model's prediction for this patient.")

    shap_values_patient = explainer_demo(patient_data)

    # Select Class 1: Alzheimer's
    shap_values_alz_patient = shap_values_patient[:, :, 1]

    # Create SHAP waterfall plot
    plt.figure()

    shap.plots.waterfall(
        shap_values_alz_patient[0],
        show=False
    )

    st.pyplot(plt.gcf())

    plt.close()


    st.caption(
        "Red features push the model prediction toward Alzheimer's, "
        "while blue features push the prediction away from Alzheimer's."
    )



# Disclaimer
st.divider()

st.caption(
    "This application is an educational demonstration based on a synthetic dataset. "
    "It is not intended for clinical diagnosis or medical decision-making."
)