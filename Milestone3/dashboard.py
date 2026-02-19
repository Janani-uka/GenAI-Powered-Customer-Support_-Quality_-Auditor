import streamlit as st
import pandas as pd

st.set_page_config(page_title="Support Audit AI", layout="wide")
st.title("📊 Support Audit AI Dashboard")

df = pd.read_csv("audit_results.csv")

st.sidebar.header("Filter")

score = st.sidebar.slider("Minimum Final Score", 0, 100, 50)

filtered_df = df[df["Final_Score_Out_of_100"] >= score]

st.subheader("Audit Results Table")
st.dataframe(filtered_df[["Clean_Text", "Final_Score_Out_of_100"]], use_container_width=True)

st.subheader("🔍 Detailed Audit")

# ✅ Important safety check
if filtered_df.empty:
    st.warning("No transcripts match this score. Reduce the filter.")
else:
    selected = st.selectbox(
        "Select Transcript",
        filtered_df["Clean_Text"]
    )

    row = filtered_df[filtered_df["Clean_Text"] == selected].iloc[0]

    st.markdown("### 📝 Transcript")
    st.write(row["Clean_Text"])

    st.markdown("### 📜 Retrieved Policy")
    st.info(row["Policy_Context"])

    st.markdown("### 🤖 LLM Audit Explanation")
    st.success(row["LLM_Audit"])
