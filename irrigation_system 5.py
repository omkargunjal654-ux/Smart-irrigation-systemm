import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.title("🌱 Smart Irrigation System")

# -------------------------------
# FILE UPLOAD (NO PATH ERROR)
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # -------------------------------
    # ENCODING (CORRECT WAY)
    # -------------------------------
    encoders = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # -------------------------------
    # SPLIT DATA
    # -------------------------------
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # TRAIN MODELS
    # -------------------------------
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)

    # -------------------------------
    # ACCURACY
    # -------------------------------
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    st.subheader("📊 Model Accuracy")
    st.write("Decision Tree:", dt_acc)
    st.write("Random Forest:", rf_acc)

    # -------------------------------
    # GRAPH
    # -------------------------------
    fig, ax = plt.subplots()
    ax.bar(["Decision Tree", "Random Forest"], [dt_acc, rf_acc])
    ax.set_title("Model Comparison")
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

    # -------------------------------
    # USER INPUT
    # -------------------------------
    st.subheader("🔍 Enter Input Values")

    user_input = []

    for col in X.columns:
        val = st.text_input(f"{col}")

        if val:
            if col in encoders:
                try:
                    val = encoders[col].transform([val])[0]
                except:
                    st.error(f"Invalid value for {col}")
                    st.stop()
            else:
                val = float(val)
            user_input.append(val)
        else:
            user_input.append(0)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    if st.button("Predict Irrigation"):
        prediction = rf_model.predict([user_input])
        st.success(f"🌿 Irrigation Decision: {prediction[0]}")