import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "model" / "churn_prediction_model.pkl"

model = joblib.load(model_path)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether an e-commerce customer is likely to churn "
    "using Machine Learning."
)

st.divider()


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=32
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    city = st.selectbox(
        "City",
        [
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Pune",
            "Hyderabad",
            "Chennai",
            "Kolkata",
            "Ahmedabad"
        ]
    )

    membership = st.selectbox(
        "Membership",
        [
            "Bronze",
            "Silver",
            "Gold",
            "Platinum"
        ]
    )


with col2:

    total_orders = st.number_input(
        "Total Orders",
        min_value=0,
        value=6
    )

    total_spent = st.number_input(
        "Total Spent",
        min_value=0.0,
        value=12000.0
    )

    avg_order_value = st.number_input(
        "Average Order Value",
        min_value=0.0,
        value=2000.0
    )

    days_since_last_purchase = st.number_input(
        "Days Since Last Purchase",
        min_value=0,
        value=85
    )


with col3:

    website_visits = st.number_input(
        "Website Visits",
        min_value=0,
        value=5
    )

    products_viewed = st.number_input(
        "Products Viewed",
        min_value=0,
        value=18
    )

    cart_abandonment_rate = st.number_input(
        "Cart Abandonment Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )

    discount_usage = st.number_input(
        "Discount Usage",
        min_value=0,
        value=5
    )


# ============================================================
# ADDITIONAL INFORMATION
# ============================================================

st.header("🛒 Additional Customer Details")

col4, col5 = st.columns(2)


with col4:

    support_tickets = st.number_input(
        "Support Tickets",
        min_value=0,
        value=2
    )


with col5:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Debit Card",
            "UPI",
            "PayPal",
            "Cash on Delivery"
        ]
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Churn",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    new_customer = pd.DataFrame({

        "Age": [age],

        "Gender": [gender],

        "City": [city],

        "Membership": [membership],

        "TotalOrders": [total_orders],

        "TotalSpent": [total_spent],

        "AvgOrderValue": [avg_order_value],

        "DaysSinceLastPurchase": [days_since_last_purchase],

        "WebsiteVisits": [website_visits],

        "ProductsViewed": [products_viewed],

        "CartAbandonmentRate": [cart_abandonment_rate],

        "DiscountUsage": [discount_usage],

        "SupportTickets": [support_tickets],

        "PaymentMethod": [payment_method]
    })


    # Make prediction
    prediction = model.predict(new_customer)[0]

    probability = model.predict_proba(new_customer)[0][1]

    probability_percent = probability * 100


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.subheader("📈 Prediction Result")


    # HIGH RISK
    if probability_percent >= 70:

        st.error(
            "🔴 HIGH CHURN RISK — Customer is likely to churn."
        )

        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )

        st.progress(probability)

        st.warning(
            "💡 Recommendation: Consider offering a personalized "
            "discount, loyalty reward, or retention campaign."
        )


    # MEDIUM RISK
    elif probability_percent >= 40:

        st.warning(
            "🟡 MEDIUM CHURN RISK — Customer may churn."
        )

        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )

        st.progress(probability)

        st.info(
            "💡 Recommendation: Increase customer engagement "
            "through personalized offers and follow-up communication."
        )


    # LOW RISK
    else:

        st.success(
            "🟢 LOW CHURN RISK — Customer is likely to stay."
        )

        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )

        st.progress(probability)

        st.success(
            "💡 Recommendation: Continue loyalty programs "
            "and maintain a positive customer experience."
        )


    # ========================================================
    # PREDICTION DETAILS
    # ========================================================

    st.divider()

    st.subheader("📋 Prediction Details")

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Prediction",
            "Churn" if prediction == 1 else "Stay"
        )


    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )


    with result_col3:

        st.metric(
            "Customer",
            "At Risk" if prediction == 1 else "Stable"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project | "
    "Python • Pandas • Scikit-learn • XGBoost • Streamlit"
)