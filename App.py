import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------- PAGE ----------------
st.set_page_config(page_title="Demand Trend Analysis", layout="centered")

# ---------------- LOAD ----------------
@st.cache_resource
def load_model():
    xgb = pickle.load(open("xgb.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    products = pickle.load(open("products.pkl", "rb"))
    return xgb, columns, products

@st.cache_data
def load_data():
    return pickle.load(open("df_grouped.pkl", "rb"))

xgb, columns, products = load_model()
df_grouped = load_data()

df_grouped["Product_lower"] = df_grouped["Product"].str.lower()

# ---------------- UI ----------------
st.title("Demand Forecasting of Products")

product = st.selectbox("Product", sorted(products))
month = st.number_input("Month (1-12)", 1, 12, 1)

discount = st.radio("Discount", ["No", "Yes"])
discount = 1 if discount == "Yes" else 0


# ---------------- PREDICT ----------------
def predict(product, month, discount):

    product = product.strip().lower()

    data = df_grouped[df_grouped["Product_lower"] == product]

    if data.empty:
        return None, None

    freq = data["Product_freq"].iloc[0]

    input_df = pd.DataFrame([{
        "Month": month,
        "Total_Cost": df_grouped["Total_Cost"].mean(),
        "Discount_Applied": discount,
        "Product_freq": freq
    }])

    input_df = input_df.reindex(columns=columns, fill_value=0)

    predicted = xgb.predict(input_df)[0]

    current_row = data[data["Month"] == month]["Total_Items"]
    current = current_row.values[0] if len(current_row) > 0 else None

    return predicted, current


# ---------------- BUTTON ----------------
if st.button("Predict Demand"):

    predicted, current = predict(product, month, discount)

    if predicted is None:
        st.error("Product not found")

    elif current is None:
        st.warning("No historical data")
        st.write("Predicted:", round(predicted))

    else:
        # ---------------- ROUND (MATCH IPYNB) ----------------
        predicted = round(predicted)
        current = round(current)

        change = predicted - current
        percent = (change / current) * 100 if current != 0 else 0

        # ---------------- TEXT OUTPUT ----------------
        st.markdown("## 📊 DEMAND TREND ANALYSIS")

        col1, col2 = st.columns(2)
        col1.metric("Current Demand", current)
        col2.metric("Predicted Demand", predicted)

        st.write("Change %:", round(percent, 2), "%")

        # ---------------- CLEAN BAR CHART ----------------
        st.markdown("### 📊 Actual vs Predicted")

        labels = ["Current", "Predicted"]
        values = [current, predicted]

        fig, ax = plt.subplots()

        bars = ax.bar(labels, values)

        # 🔥 keep chart clean (no big difference feel)
        ax.set_ylim(min(values) * 0.95, max(values) * 1.05)

        ax.set_ylabel("Demand")
        ax.set_title("Actual vs Predicted Demand")

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    str(height), ha='center', va='bottom')

        st.pyplot(fig)
        if change > 0:
            st.success(f"📈 Demand Increased by: {change} units")
        elif change < 0:
            st.warning(f"📉 Demand Decreased by: {abs(change)} units")
        else:
            st.info("➡️ Demand Unchanged")