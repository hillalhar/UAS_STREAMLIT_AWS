"""
Streamlit app untuk Credit Score Classification.
Memanggil SageMaker endpoint yang sudah di-deploy lewat boto3.

boto3 membaca kredensial AWS dari:
  - EC2 instance profile (LabInstanceProfile) jika berjalan di EC2/SageMaker notebook, atau
  - ~/.aws/credentials jika berjalan secara lokal

Jalankan dengan:
    streamlit run app_streamlit_aws.py

atau set environment variable jika endpoint/region berbeda:
    ENDPOINT_NAME=credit-score-endpoint AWS_REGION=us-east-1 streamlit run app_streamlit_aws.py
"""

import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: dict) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.set_page_config(page_title="Credit Score App", page_icon="💳", layout="centered")
st.title("💳 Prediksi Credit Score")
st.write(
    "Isi profil finansial nasabah di bawah untuk memprediksi kategori Credit Score "
    "(**Poor**, **Standard**, atau **Good**) melalui SageMaker endpoint."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 25)
        annual_income = st.number_input("Annual Income", value=50000.0)
        salary = st.number_input("Monthly Inhand Salary", value=4000.0)
        num_bank_accounts = st.number_input("Num Bank Accounts", value=1)
        num_credit_card = st.number_input("Num Credit Cards", value=1)
        interest_rate = st.number_input("Interest Rate (%)", value=5)
        num_loans = st.number_input("Num Loans", value=1)
        total_emi = st.number_input("Total EMI per month", value=100.0)
        credit_util = st.number_input("Credit Utilization Ratio", value=30.0)
        monthly_bal = st.number_input("Monthly Balance", value=500.0)
        delay_days = st.number_input("Delay from Due Date", value=0)

    with col2:
        num_delayed = st.number_input("Num Delayed Payments", value=0)
        credit_history = st.number_input("Credit History Age (Months)", value=12.0)
        credit_inquiries = st.number_input("Credit Inquiries", value=1.0)
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
        min_payment = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"])
        payment_behaviour = st.selectbox(
            "Payment Behaviour",
            [
                "Low_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments",
                "High_spent_Small_value_payments",
            ],
        )
        occupation = st.text_input("Occupation", "Engineer")
        month = st.selectbox(
            "Month",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"],
        )
        changed_limit = st.number_input("Changed Credit Limit", value=10.0)
        outstanding_debt = st.number_input("Outstanding Debt", value=500.0)
        amount_invested = st.number_input("Amount Invested Monthly", value=100.0)

    submit = st.form_submit_button("🔍 Prediksi", use_container_width=True)

if submit:
    features = {
        "Age": age, "Annual_Income": annual_income, "Monthly_Inhand_Salary": salary,
        "Num_Bank_Accounts": num_bank_accounts, "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate, "Num_of_Loan": num_loans,
        "Delay_from_due_date": delay_days, "Num_of_Delayed_Payment": num_delayed,
        "Changed_Credit_Limit": changed_limit, "Num_Credit_Inquiries": credit_inquiries,
        "Outstanding_Debt": outstanding_debt, "Credit_Utilization_Ratio": credit_util,
        "Credit_History_Age": credit_history, "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": amount_invested, "Monthly_Balance": monthly_bal,
        "Occupation": occupation, "Payment_of_Min_Amount": min_payment,
        "Payment_Behaviour": payment_behaviour, "Credit_Mix": credit_mix, "Month": month,
    }

    try:
        result = invoke_endpoint(features)

        st.divider()
        st.subheader("Hasil Prediksi")

        label = result["predictions"][0]
        label_color = {"Good": "green", "Standard": "orange", "Poor": "red"}.get(label, "gray")
        st.markdown(f"### Credit Score: :{label_color}[**{label}**]")

        if "probabilities" in result:
            st.write("Probabilitas tiap kelas:")
            probs = result["probabilities"][0]
            for cls in ["Poor", "Standard", "Good"]:
                st.progress(probs[cls], text=f"{cls}: {probs[cls]*100:.1f}%")

    except NoCredentialsError:
        st.error(
            "AWS credentials tidak ditemukan. "
            "Jika di EC2/SageMaker, pastikan LabInstanceProfile sudah di-attach. "
            "Jika lokal, konfigurasi ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    except Exception as e:
        st.error(f"Error: {e}")
