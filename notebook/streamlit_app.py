import os
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Marketplace Intelligence Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Marketplace Intelligence Assistant")
st.caption("AI-powered executive insights using marketplace KPIs, customer analytics, operations data, and ML results")


# -----------------------------
# File paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    kpis = pd.read_csv(OUTPUT_DIR / "executive_kpis.csv")
    importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv").head(10)
    monthly = pd.read_csv(OUTPUT_DIR / "monthly_revenue.csv")
    rfm = pd.read_csv(OUTPUT_DIR / "rfm_summary.csv")
    operations = pd.read_csv(OUTPUT_DIR / "operations_performance.csv")
    return kpis, importance, monthly, rfm, operations


kpis, importance, monthly, rfm, operations = load_data()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Example Questions")
    st.markdown("""
- Summarize marketplace performance.
- What are the biggest risks to customer satisfaction?
- Which factors predict poor reviews?
- What should operations improve first?
- Which customer segment is most valuable?
- Give me 5 business recommendations.
- What are the executive KPIs?
""")


# -----------------------------
# KPI cards
# -----------------------------
kpi_dict = dict(zip(kpis["KPI"], kpis["Value"]))

c1, c2, c3, c4 = st.columns(4)

c1.metric("Revenue", f"${float(kpi_dict['Gross Revenue']):,.0f}")
c2.metric("Orders", f"{int(kpi_dict['Total Orders']):,}")
c3.metric("Customers", f"{int(kpi_dict['Total Customers']):,}")
c4.metric("AOV", f"${float(kpi_dict['Average Order Value']):.2f}")

st.divider()


# -----------------------------
# Main assistant
# -----------------------------
st.subheader("Ask the Executive Copilot")

question = st.text_input("Ask a business question")

api_key = os.getenv("OPENAI_API_KEY")

if st.button("Ask") and question.strip():

    if not api_key:
        st.error("OPENAI_API_KEY not found. Set it in your terminal before running Streamlit.")
        st.code('setx OPENAI_API_KEY "your_api_key_here"', language="powershell")
        st.stop()

    client = OpenAI(api_key=api_key)

    context = f"""
Executive KPIs:
{kpis.to_string(index=False)}

Monthly Revenue - Latest 12 Rows:
{monthly.tail(12).to_string(index=False)}

Customer Segments - RFM Summary:
{rfm.to_string(index=False)}

Operations Performance:
{operations.to_string(index=False)}

Top ML Feature Importance:
{importance.to_string(index=False)}
"""

    with st.spinner("Analyzing marketplace data..."):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
You are the Chief Analytics Officer of a marketplace company.

Answer only using the provided analytics data.

Always structure your response as:

## Executive Summary

## Evidence from Data

## Business Recommendations

If the answer cannot be supported by the provided analytics, say so clearly.
"""
                },
                {
                    "role": "user",
                    "content": context + "\n\nBusiness Question: " + question
                }
            ],
            temperature=0.2
        )

    st.success("Analysis complete")
    st.markdown(response.choices[0].message.content)


# -----------------------------
# Preview tables
# -----------------------------
with st.expander("About this Analysis"):
    st.subheader("Executive KPIs")
    st.dataframe(kpis, use_container_width=True)

    st.subheader("Top ML Features")
    st.dataframe(importance, use_container_width=True)

    st.subheader("Customer Segments")
    st.dataframe(rfm, use_container_width=True)