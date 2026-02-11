import streamlit as st
import pandas as pd
import openai
import os
import plotly.express as px
from dotenv import load_dotenv
import io
import time

# --------- SETUP SECTION ---------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=OPENAI_API_KEY)

# --------- SESSION STATE FOR AI TELEMETRY HISTORY ---------
if "ai_telemetry_history" not in st.session_state:
    st.session_state.ai_telemetry_history = []

# --------- AI SUGGESTION FUNCTION WITH PROCEDURAL & MEDICAL ---------
def ai_agentic_recommendation(patient_info, abnormalities):
    """
    Calls OpenAI API to get both procedural and medical suggestions for ICU nurse actions.
    Returns procedural and medical recommendations, plus telemetry.
    """
    prompt = f"""
    Patient Info: {patient_info}
    Abnormal Findings: {abnormalities}
    As an ICU Nurse Agent, provide:
    1. Procedural recommendations: List step-by-step nursing and escalation actions (e.g., notify clinician, verify readings, activate rapid response, document, etc.).
    2. Medical recommendations: List possible differential diagnoses and general medical considerations or interventions that a clinician might consider, based on the abnormalities. Do not give specific drug names or dosages. Add a disclaimer that this is for educational demonstration only and not real medical advice.
    Format your response as two lists titled "Procedural Recommendations" and "Medical Recommendations".
    """
    telemetry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "latency_seconds": None,
        "abnormalities": abnormalities.copy() if abnormalities else [],
    }
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are an ICU nursing assistant AI. You provide both procedural and high-level medical recommendations for educational purposes. "
                    "Never provide real medical advice, drug names, or dosages. Always include a disclaimer for medical information."
                )},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2,
        )
        latency = time.time() - start_time
        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        if usage:
            telemetry["input_tokens"] = usage.prompt_tokens
            telemetry["output_tokens"] = usage.completion_tokens
            telemetry["total_tokens"] = usage.total_tokens
        telemetry["latency_seconds"] = latency

        # Append telemetry to session history
        st.session_state.ai_telemetry_history.append(telemetry)

        # Parse response into two lists
        procedural, medical = [], []
        in_proc, in_med = False, False
        for line in content.splitlines():
            if "Procedural Recommendations" in line:
                in_proc, in_med = True, False
                continue
            if "Medical Recommendations" in line:
                in_proc, in_med = False, True
                continue
            if in_proc and line.strip().startswith(("-", "*", "1.")):
                procedural.append(line.lstrip("-*0123456789. ").strip())
            elif in_med and line.strip().startswith(("-", "*", "1.")):
                medical.append(line.lstrip("-*0123456789. ").strip())
        # Fallback if parsing fails
        if not procedural and not medical:
            procedural = [content]
        return procedural, medical, telemetry
    except Exception as e:
        st.session_state.ai_telemetry_history.append(telemetry)
        return [f"AI recommendation error: {str(e)}"], [], telemetry

# --------- ABNORMALITY DETECTION ---------
def detect_abnormalities(df):
    abnormalities = []
    if 'heart_rate_bpm' in df.columns:
        if (df['heart_rate_bpm'] < 50).any() or (df['heart_rate_bpm'] > 120).any():
            abnormalities.append('Abnormal Heart Rate')
    if 'bp_systolic_mmHg' in df.columns:
        if (df['bp_systolic_mmHg'] < 90).any() or (df['bp_systolic_mmHg'] > 180).any():
            abnormalities.append('Abnormal Systolic BP')
    if 'spo2_percent' in df.columns:
        if (df['spo2_percent'] < 92).any():
            abnormalities.append('Low Oxygen Saturation')
    return abnormalities

# --------- EMERGENCY FLASH ALERT ---------
def flash_emergency_alert(abnormalities):
    emergency_map = {
        'Abnormal Heart Rate': "⚠️ EMERGENCY: Abnormal Heart Rate Detected! IMMEDIATE ATTENTION REQUIRED!",
        'Low Oxygen Saturation': "⚠️ EMERGENCY: Low Oxygen Saturation Detected! IMMEDIATE ATTENTION REQUIRED!",
        'Abnormal Systolic BP': "⚠️ EMERGENCY: Abnormal Systolic Blood Pressure Detected! IMMEDIATE ATTENTION REQUIRED!"
    }
    for abnormality in abnormalities:
        if abnormality in emergency_map:
            st.markdown(
                f"<h2 style='color:red; animation: blinker 1s linear infinite;'>{emergency_map[abnormality]}</h2>"
                "<style>@keyframes blinker { 50% { opacity: 0; } }</style>",
                unsafe_allow_html=True
            )

# --------- PATIENT TRENDS VISUALIZATION ---------
def plot_trends(df):
    st.subheader("Patient Vitals Trends")
    numeric_cols = ['heart_rate_bpm', 'temperature_c', 'bp_systolic_mmHg', 'bp_diastolic_mmHg', 'spo2_percent']
    time_col = 'timestamp'
    for col in numeric_cols:
        if col in df.columns:
            fig = px.line(df, x=time_col, y=col, title=f"{col.replace('_',' ').title()} Over Time")
            st.plotly_chart(fig, use_container_width=True)

# --------- SUMMARY STATISTICS ---------
def show_statistics(df):
    st.subheader("Summary Statistics")
    stats_df = df[['heart_rate_bpm', 'temperature_c', 'bp_systolic_mmHg', 'bp_diastolic_mmHg', 'spo2_percent']].agg(['min', 'max', 'mean'])
    st.dataframe(stats_df)

# --------- CONDITIONAL FORMATTING FOR TIMELINE ---------
def highlight_critical(row):
    color = ''
    if row['heart_rate_bpm'] > 120 or row['spo2_percent'] < 92:
        color = 'background-color: red; color: white'
    return [color]*len(row)

def show_timeline(df):
    st.subheader("Patient Timeline (Critical Values Highlighted)")
    styled_df = df.style.apply(highlight_critical, axis=1)
    st.dataframe(styled_df)

# --------- FILTERING FUNCTION ---------
def filter_data(df):
    st.subheader("Filter Timeline")
    times = df['timestamp'].unique()
    selected_times = st.multiselect("Select timestamps to view", times)
    if selected_times:
        filtered_df = df[df['timestamp'].isin(selected_times)]
        st.dataframe(filtered_df)
    else:
        st.dataframe(df)

# --------- DOWNLOADABLE REPORTS ---------
def download_report(df, abnormalities, procedural, medical):
    st.subheader("Download Reports")
    report = f"Abnormalities Detected: {', '.join(abnormalities)}\n"
    report += "Procedural Recommendations:\n" + "\n".join(procedural)
    report += "\nMedical Recommendations (Educational Only):\n" + "\n".join(medical)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Patient Data CSV",
        data=csv_buffer.getvalue(),
        file_name="patient_data_report.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Recommendations TXT",
        data=report,
        file_name="recommendations.txt",
        mime="text/plain"
    )

# --------- LATEST CONDITION DISPLAY ---------
def show_latest(df):
    latest = df.iloc[-1]
    st.subheader("Latest Patient Condition")
    cols = st.columns(4)
    cols[0].metric("Heart Rate (bpm)", latest['heart_rate_bpm'])
    cols[1].metric("Systolic BP (mmHg)", latest['bp_systolic_mmHg'])
    cols[2].metric("SpO2 (%)", latest['spo2_percent'])
    cols[3].metric("Temp (°C)", latest['temperature_c'])

# --------- ADVANCED AI OBSERVABILITY VIEW ---------
def show_ai_observability():
    st.subheader("AI Observability Dashboard")
    telemetry_history = st.session_state.ai_telemetry_history
    if not telemetry_history or all(t["input_tokens"] is None for t in telemetry_history):
        st.info("No AI telemetry data available yet. Run at least one analysis to populate observability metrics.")
        return
    df_telemetry = pd.DataFrame(telemetry_history)
    st.dataframe(df_telemetry)

    st.markdown("#### Token Usage Over Time")
    token_cols = ['input_tokens', 'output_tokens', 'total_tokens']
    for col in token_cols:
        if df_telemetry[col].notnull().any():
            fig = px.line(df_telemetry, x="timestamp", y=col, title=f"{col.replace('_',' ').title()} Over Time")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### API Latency Over Time")
    if df_telemetry['latency_seconds'].notnull().any():
        fig = px.line(df_telemetry, x="timestamp", y="latency_seconds", title="API Latency (seconds) Over Time")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Latest API Call")
    latest = df_telemetry.iloc[-1]
    st.markdown(
        f"""
        - **Timestamp:** {latest['timestamp']}
        - **Input tokens:** {latest['input_tokens']}
        - **Output tokens:** {latest['output_tokens']}
        - **Total tokens:** {latest['total_tokens']}
        - **API latency:** {latest['latency_seconds']:.2f} seconds
        - **Abnormalities (if any):** {', '.join(latest['abnormalities']) if latest['abnormalities'] else 'None'}
        """
    )

# --------- MAIN APP FUNCTION ---------
def main():
    st.set_page_config(page_title="Agentic ICU Patient Monitor", layout="wide")
    st.title("Agentic ICU Patient Monitoring App (Academic Demonstration)")
    st.info(
        "Upload a CSV file with patient data to begin analysis. "
        "This app will visualize trends, highlight abnormalities, and provide both procedural and medical AI recommendations for demonstration purposes.\n\n"
        "**Disclaimer:** This app is for academic demonstration only. Not for real clinical use."
    )

    view = st.sidebar.radio("Select View", [
        "Overview & Trends", 
        "Summary Stats", 
        "Timeline (Criticals Highlighted)", 
        "Filter Timeline", 
        "Recommendations", 
        "Download Reports",
        "AI Observability (Telemetry)"
    ])
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Expected CSV Columns:**")
    st.sidebar.markdown(
        "- patient_id\n"
        "- timestamp\n"
        "- ECG\n"
        "- heart_rate_bpm\n"
        "- temperature_c\n"
        "- bp_systolic_mmHg\n"
        "- bp_diastolic_mmHg\n"
        "- spo2_percent"
    )

    uploaded_file = st.file_uploader("Upload Patient CSV File", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        for col in ['heart_rate_bpm', 'temperature_c', 'bp_systolic_mmHg', 'bp_diastolic_mmHg', 'spo2_percent']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        abnormalities = detect_abnormalities(df)
        procedural, medical, ai_telemetry = ai_agentic_recommendation(df.to_dict(), abnormalities)
        flash_emergency_alert(abnormalities)

        if view == "Overview & Trends":
            show_latest(df)
            plot_trends(df)
        elif view == "Summary Stats":
            show_statistics(df)
        elif view == "Timeline (Criticals Highlighted)":
            show_timeline(df)
        elif view == "Filter Timeline":
            filter_data(df)
        elif view == "Recommendations":
            st.subheader("Detected Abnormalities")
            if abnormalities:
                st.warning(f"{', '.join(abnormalities)}")
            else:
                st.success("No significant abnormalities detected in the uploaded data.")

            st.subheader("Procedural Recommendations (Agentic AI):")
            for rec in procedural:
                st.write(f"- {rec}")

            st.subheader("Medical Recommendations (Agentic AI, Educational Only):")
            for rec in medical:
                st.write(f"- {rec}")
            st.caption("These medical recommendations are for educational demonstration only. They do not constitute medical advice. Always consult a qualified clinician.")

        elif view == "Download Reports":
            download_report(df, abnormalities, procedural, medical)
        elif view == "AI Observability (Telemetry)":
            show_ai_observability()

        st.markdown("---")
        st.subheader("Raw Patient Data")
        st.dataframe(df)
    else:
        if view == "AI Observability (Telemetry)":
            show_ai_observability()
        else:
            st.info("Please upload a patient CSV file to begin.")

if __name__ == "__main__":
    main()