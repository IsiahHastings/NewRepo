import streamlit as st
import pandas as pd
import joblib
import time
from streamlit_autorefresh import st_autorefresh

# ==========================================
# Configuration
# ==========================================

st.set_page_config(
    page_title="Generator Health Monitor",
    layout="wide"
)

st.title("Generator Health Monitoring Dashboard 📈")

count = st_autorefresh(
    interval=5000,
    key="generator_refresh"
)

# ==========================================
# Load Models
# ==========================================

@st.cache_resource
def load_models():

    isolation_model = joblib.load("isolation_forest.pkl")
    svm_model = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")

    return isolation_model, svm_model, scaler


isolation_model, svm_model, scaler = load_models()


# ==========================================
# Placeholders
# ==========================================

status_box = st.empty()

metrics_area = st.empty()

charts_area = st.empty()


# ==========================================
# Main Update
# ==========================================

def update_dashboard():

    try:

        data = pd.read_csv("live_data.csv")


        if len(data) < 2:
            st.warning("Waiting for MATLAB data...")
            return


        latest = data.iloc[-1]


        Vrms = latest["Vrms"]
        Irms = latest["Irms"]
        Temperature = latest["Temperature"]
        Vibration = latest["Vibration"]


        # -------------------------------
        # Display measurements
        # -------------------------------

        with metrics_area.container():

            st.subheader("Live Generator Measurements")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Voltage RMS",
                f"{Vrms:.2f} V"
            )

            c2.metric(
                "Current RMS",
                f"{Irms:.2f} A"
            )

            c3.metric(
                "Temperature",
                f"{Temperature:.2f} °C"
            )

            c4.metric(
                "Vibration",
                f"{Vibration:.2f}"
            )


        # -------------------------------
        # AI Prediction
        # -------------------------------

        X = pd.DataFrame(
            [[Vrms, Irms, Temperature, Vibration]],
            columns=[
                "Vrms",
                "Irms",
                "Temperature",
                "Vibration"
            ]
        )


        anomaly = isolation_model.predict(X)[0]


        with status_box.container():

            st.subheader("AI Diagnosis")


            if anomaly == 1:

                st.success(
                    "🟢 NORMAL OPERATION"
                )


            else:

                st.error(
                    "🔴 ANOMALY DETECTED"
                )


                X_scaled = scaler.transform(X)

                fault = svm_model.predict(
                    X_scaled
                )[0]


                st.warning(
                    f"Fault Type: {fault}"
                )


        # -------------------------------
        # Graphs
        # -------------------------------

        with charts_area.container():

            st.subheader("Live Trends")


            st.line_chart(
                data.set_index("Time")[["Vrms"]]
            )


            st.line_chart(
                data.set_index("Time")[["Irms"]]
            )


            st.line_chart(
                data.set_index("Time")[["Temperature"]]
            )


            st.line_chart(
                data.set_index("Time")[["Vibration"]]
            )


    except Exception as e:

        st.error(
            f"Error: {e}"
        )


update_dashboard()


# Refresh every 5 seconds

#time.sleep(5)

#st.rerun()