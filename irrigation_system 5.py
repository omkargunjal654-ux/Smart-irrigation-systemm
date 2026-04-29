import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Irrigation System", page_icon="🌱")
st.title("🌱 Smart Irrigation System")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # -----------------------------------------------
    # ENCODING — encode EVERY column without exception
    # -----------------------------------------------
    encoders = {}
    df_encoded = df.copy()

    for col in df_encoded.columns:
        le = LabelEncoder()
        # Convert to string first to handle ANY dtype (int, float, object, StringDtype)
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # -----------------------------------------------
    # SPLIT — use .values to get pure numpy arrays
    # -----------------------------------------------
    X = df_encoded.iloc[:, :-1].values.astype(float)
    y = df_encoded.iloc[:, -1].values.astype(float)

    feature_names = df.columns[:-1].tolist()
    target_col    = df.columns[-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------------------------
    # TRAIN
    # -----------------------------------------------
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # -----------------------------------------------
    # ACCURACY
    # -----------------------------------------------
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    st.subheader("📊 Model Accuracy")
    c1, c2 = st.columns(2)
    c1.metric("🌿 Decision Tree", f"{dt_acc * 100:.2f}%")
    c2.metric("🌲 Random Forest", f"{rf_acc * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Decision Tree", "Random Forest"], [dt_acc, rf_acc],
                  color=["#4CAF50", "#2196F3"], width=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2%}", ha='center', fontweight='bold')
    st.pyplot(fig)

    # -----------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------
    st.subheader("🌟 Feature Importance (Random Forest)")
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(imp_df["Feature"], imp_df["Importance"], color="#4CAF50")
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Feature Importance")
    ax2.invert_yaxis()
    st.pyplot(fig2)

    # -----------------------------------------------
    # USER INPUT — dropdowns for text, number for numeric
    # -----------------------------------------------
    st.subheader("🔍 Enter Input Values for Prediction")

    user_input = []
    grid = st.columns(3)

    for i, col in enumerate(feature_names):
        with grid[i % 3]:
            original_vals = df[col].astype(str).unique().tolist()
            # If original column had non-numeric values → show dropdown
            try:
                [float(v) for v in original_vals]
                is_text = False
            except ValueError:
                is_text = True

            if is_text:
                chosen = st.selectbox(col, sorted(original_vals), key=f"inp_{col}")
                encoded_val = float(encoders[col].transform([str(chosen)])[0])
            else:
                encoded_val = st.number_input(
                    col,
                    value=float(df[col].astype(float).mean()),
                    key=f"inp_{col}"
                )
            user_input.append(encoded_val)

    # -----------------------------------------------
    # PREDICTION
    # -----------------------------------------------
    if st.button("🌿 Predict Irrigation"):
        input_array = np.array(user_input, dtype=float).reshape(1, -1)

        dt_pred_enc = int(dt_model.predict(input_array)[0])
        rf_pred_enc = int(rf_model.predict(input_array)[0])

        dt_pred = encoders[target_col].inverse_transform([dt_pred_enc])[0]
        rf_pred = encoders[target_col].inverse_transform([rf_pred_enc])[0]

        st.success(f"🌿 Decision Tree says: **{dt_pred}**")
        st.success(f"🌲 Random Forest says: **{rf_pred}**")

        if str(rf_pred).upper() == "ON":
            st.info("💧 Recommendation: **Irrigate the field**")
        else:
            st.info("✅ Recommendation: **No irrigation needed**")

else:
    st.info("👆 Please upload the irrigation CSV dataset to get started.")
    st.markdown("""
    **Expected columns:**
    `Temparature`, `Humidity`, `Moisture`, `Soil Type`, `Crop Type`,
    `Nitrogen`, `Potassium`, `Phosphorous`, `Fertilizer Name`, `Irrigation`
    """)
