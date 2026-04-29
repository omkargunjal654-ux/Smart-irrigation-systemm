import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Irrigation System", page_icon="🌱")
st.title("🌱 Smart Irrigation System")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # -------------------------------
    # ENCODING
    # -------------------------------
    encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le

    # -------------------------------
    # SPLIT DATA
    # -------------------------------
    X = df_encoded.iloc[:, :-1]
    y = df_encoded.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # TRAIN MODELS
    # -------------------------------
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # -------------------------------
    # ACCURACY
    # -------------------------------
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    st.subheader("📊 Model Accuracy")
    col1, col2 = st.columns(2)
    col1.metric("Decision Tree", f"{dt_acc * 100:.2f}%")
    col2.metric("Random Forest", f"{rf_acc * 100:.2f}%")

    # -------------------------------
    # ACCURACY CHART
    # -------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Decision Tree", "Random Forest"],
        [dt_acc, rf_acc],
        color=["#4CAF50", "#2196F3"],
        width=0.4
    )
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2%}",
            ha='center', va='bottom', fontweight='bold'
        )
    st.pyplot(fig)

    # -------------------------------
    # FEATURE IMPORTANCE (Random Forest)
    # -------------------------------
    st.subheader("🌟 Feature Importance (Random Forest)")
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(importance_df["Feature"], importance_df["Importance"], color="#4CAF50")
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Feature Importance")
    ax2.invert_yaxis()
    st.pyplot(fig2)

    # -------------------------------
    # USER INPUT FOR PREDICTION
    # -------------------------------
    st.subheader("🔍 Enter Input Values for Prediction")

    user_input = []
    valid = True

    cols = st.columns(min(3, len(X.columns)))
    for i, col in enumerate(X.columns):
        with cols[i % len(cols)]:
            val = st.text_input(f"{col}", key=col)
            if val:
                if col in encoders:
                    if val in encoders[col].classes_:
                        val = encoders[col].transform([val])[0]
                    else:
                        st.error(f"❌ Invalid value for '{col}'. Valid options: {list(encoders[col].classes_)}")
                        valid = False
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        st.error(f"❌ '{col}' must be a number.")
                        valid = False
                user_input.append(val)
            else:
                user_input.append(0)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    if st.button("🌿 Predict Irrigation"):
        if not valid:
            st.warning("⚠️ Please fix the errors above before predicting.")
        else:
            dt_pred = dt_model.predict([user_input])[0]
            rf_pred = rf_model.predict([user_input])[0]

            # Decode label if encoded
            target_col = df.columns[-1]
            if target_col in encoders:
                dt_pred = encoders[target_col].inverse_transform([dt_pred])[0]
                rf_pred = encoders[target_col].inverse_transform([rf_pred])[0]

            st.success(f"🌿 **Decision Tree** says: `{dt_pred}`")
            st.success(f"🌲 **Random Forest** says: `{rf_pred}`")

else:
    st.info("👆 Please upload a CSV file to get started.")
    st.markdown("""
    **Expected CSV format:**
    - All columns except the last are **features** (e.g., soil moisture, temperature, humidity)
    - The **last column** is the **target** (e.g., Irrigate / Don't Irrigate)
    """)
