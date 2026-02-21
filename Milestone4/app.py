import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import tempfile

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Support Audit Dashboard", layout="wide")

st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #f4f6f9;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='color:#1f4e79;'>📊 Support Audit AI Dashboard</h2>", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("Milestone4/audit_results.csv")

# Rename score column
if "Final_Score_Out_of_100" in df.columns:
    df.rename(columns={"Final_Score_Out_of_100": "Final_Score"}, inplace=True)

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# -------------------------------------------------
# KPI METRICS
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Empathy", round(df["Empathy"].mean(), 2))
col2.metric("Professionalism", round(df["Professionalism"].mean(), 2))
col3.metric("Compliance", round(df["Compliance"].mean(), 2))
col4.metric("Final Score", round(df["Final_Score"].mean(), 2))

st.divider()

# -------------------------------------------------
# AGENT PERFORMANCE CHART
# -------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

agent_perf = df.groupby("Agent_Name")[["Empathy","Professionalism","Compliance","Final_Score"]].mean().reset_index()

fig1 = px.bar(
    agent_perf,
    x="Agent_Name",
    y="Final_Score",
    color="Final_Score",
    color_continuous_scale="Blues",
    title="Agent-wise Performance",
)

fig1.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
chart_col1.plotly_chart(fig1, use_container_width=True)

# -------------------------------------------------
# COMPLIANCE TREND BY DATE
# -------------------------------------------------
trend = df.groupby("Date")["Compliance"].mean().reset_index()

fig2 = px.line(
    trend,
    x="Date",
    y="Compliance",
    markers=True,
    title="Compliance Trend Over Time"
)

fig2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
chart_col2.plotly_chart(fig2, use_container_width=True)

st.divider()

# -------------------------------------------------
# COMPLIANCE ALERT
# -------------------------------------------------
st.subheader("🚨 Compliance Alerts")

low_compliance = df[df["Compliance"] < 1]

if not low_compliance.empty:
    agents = low_compliance["Agent_Name"].unique()
    st.error(f"⚠ Compliance violation detected for: {', '.join(agents)}")
else:
    st.success("✅ All agents are fully compliant.")

st.divider()

# -------------------------------------------------
# AI SUGGESTIONS
# -------------------------------------------------
st.subheader("💡 AI Performance Suggestions")

avg_score = df["Final_Score"].mean()

if avg_score < 70:
    st.info("""
• Empathy improvement training is recommended to enhance customer interaction quality.  
• Strengthen compliance adherence to prevent policy violations.  
• Conduct periodic performance reviews for continuous improvement.
""")
else:
    st.success("""
• Overall performance is satisfactory.  
• Continue regular monitoring and refresher training.  
• Maintain high compliance standards.
""")

st.divider()

# -------------------------------------------------
# EXPORT TO EXCEL
# -------------------------------------------------
def convert_to_excel(dataframe):
    output = BytesIO()
    dataframe.to_excel(output, index=False, engine='openpyxl')
    return output.getvalue()

excel_data = convert_to_excel(df)

st.download_button(
    label="📥 Download Excel Report",
    data=excel_data,
    file_name="Support_Audit_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# -------------------------------------------------
# PDF REPORT
# -------------------------------------------------
def generate_pdf():
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Support Audit AI Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Average Empathy: {round(df['Empathy'].mean(),2)}", styles["Normal"]))
    elements.append(Paragraph(f"Average Professionalism: {round(df['Professionalism'].mean(),2)}", styles["Normal"]))
    elements.append(Paragraph(f"Average Compliance: {round(df['Compliance'].mean(),2)}", styles["Normal"]))
    elements.append(Paragraph(f"Average Final Score: {round(df['Final_Score'].mean(),2)}", styles["Normal"]))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("<b>Key Recommendations:</b>", styles["Heading2"]))
    elements.append(Paragraph("Empathy training and stricter compliance monitoring are recommended.", styles["Normal"]))

    if not low_compliance.empty:
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("<b>Compliance Alert:</b>", styles["Heading2"]))
        elements.append(Paragraph(f"Violation detected for: {', '.join(agents)}", styles["Normal"]))

    doc.build(elements)
    return temp_file.name

if st.button("📄 Generate PDF Report"):
    pdf_path = generate_pdf()
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download PDF",
            data=f,
            file_name="Support_Audit_Report.pdf",
            mime="application/pdf"
        )