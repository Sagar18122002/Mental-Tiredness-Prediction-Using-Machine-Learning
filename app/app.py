import streamlit as st
import pandas as pd
import sys
import os
from streamlit_extras.let_it_rain import rain
# ---------------- Add src folder ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from prediction import predict

# ---------------- Load Dataset ---------------- #

df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "mental_tiredness_score_prediction_dataset.csv.csv"
    )
)

# ---------------- Page Config ---------------- #

st.set_page_config(
    page_title="Mental Tiredness Prediction",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Mental Tiredness Prediction System")

st.markdown(
"""
Predict a person's **Mental Tiredness Score** based on
their workload, sleep habits, environment and lifestyle.
"""
)

st.markdown("---")

# ================== INPUTS =================== #

col1, col2 = st.columns(2)

with col1:

    mood = st.selectbox(
        "Mood",
        ["Low", "Neutral", "Happy"]
    )

    work_environment = st.selectbox(
        "Work Environment",
        ["Quiet", "Moderate Noise", "Noisy"]
    )

    work_type = st.selectbox(
        "Work Type",
        ["Office", "Remote", "Manual", "Student"]
    )

    number_of_decisions_made = st.slider(
        "Number of Decisions Made",
        0,
        int(df["number_of_decisions_made"].max()),
        10
    )

    context_switch_count = st.slider(
        "Context Switch Count",
        0,
        int(df["context_switch_count"].max()),
        10
    )

    notifications_received = st.slider(
        "Notifications Received",
        0,
        int(df["notifications_received"].max()),
        20
    )

    screen_time_min = st.slider(
        "Screen Time (Minutes)",
        0,
        int(df["screen_time_min"].max()),
        300
    )

    deep_work_min = st.slider(
        "Deep Work (Minutes)",
        0,
        int(df["deep_work_min"].max()),
        180
    )

with col2:

    task_complexity_avg = st.slider(
        "Task Complexity Average",
        float(df["task_complexity_avg"].min()),
        float(df["task_complexity_avg"].max()),
        5.0
    )

    caffeine_mg = st.slider(
        "Caffeine Intake (mg)",
        0,
        int(df["caffeine_mg"].max()),
        150
    )

    break_frequency = st.slider(
        "Break Frequency",
        0,
        int(df["break_frequency"].max()),
        5
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        float(df["sleep_hours"].min()),
        float(df["sleep_hours"].max()),
        7.0
    )

    deep_sleep_pct = st.slider(
        "Deep Sleep Percentage",
        float(df["deep_sleep_pct"].min()),
        float(df["deep_sleep_pct"].max()),
        80.0
    )

    hydration_l = st.slider(
        "Hydration (Litres)",
        float(df["hydration_l"].min()),
        float(df["hydration_l"].max()),
        2.5
    )

    noise_level_db = st.slider(
        "Noise Level (dB)",
        float(df["noise_level_db"].min()),
        float(df["noise_level_db"].max()),
        50.0
    )

    temperature_c = st.slider(
        "Temperature (°C)",
        float(df["temperature_c"].min()),
        float(df["temperature_c"].max()),
        24.0
    )

    workload_score = st.slider(
        "Workload Score",
        float(df["workload_score"].min()),
        float(df["workload_score"].max()),
        75.0
    )

st.markdown("---")
# ================= Prediction ================= #

if st.button("🔍 Predict Mental Tiredness Score", use_container_width=True):

    sample = {
        "mood": mood,
        "work_environment": work_environment,
        "work_type": work_type,
        "number_of_decisions_made": number_of_decisions_made,
        "context_switch_count": context_switch_count,
        "notifications_received": notifications_received,
        "screen_time_min": screen_time_min,
        "deep_work_min": deep_work_min,
        "task_complexity_avg": task_complexity_avg,
        "caffeine_mg": caffeine_mg,
        "break_frequency": break_frequency,
        "sleep_hours": sleep_hours,
        "deep_sleep_pct": deep_sleep_pct,
        "hydration_l": hydration_l,
        "noise_level_db": noise_level_db,
        "temperature_c": temperature_c,
        "workload_score": workload_score
    }

    prediction = predict(sample)

    st.markdown("---")

    st.metric(
        label="🧠 Predicted Mental Tiredness Score",
        value=f"{prediction:.2f}/100"
    )

    st.progress(min(int(prediction), 100))

    # ================= Result ================= #

    if prediction <= 20:

        st.balloons()

        try:
            rain(
                emoji="🎉",
                font_size=45,
                falling_speed=5,
                animation_length="4"
            )
        except:
            pass

        st.success("🟢 Very Low Mental Tiredness")

        st.write("""
### 😊 Recommendation
- Excellent mental health
- Keep maintaining your routine
- Continue proper sleep
- Stay hydrated
- Great work-life balance
""")

    elif prediction <= 40:

        st.balloons()

        try:
            rain(
                emoji="😊",
                font_size=45,
                falling_speed=5,
                animation_length="4"
            )
        except:
            pass

        st.success("🟢 Low Mental Tiredness")

        st.write("""
### 🙂 Recommendation
- Mental tiredness is low
- Continue healthy habits
- Take short breaks during work
- Maintain proper hydration
""")

    elif prediction <= 60:

        st.snow()

        try:
            rain(
                emoji="😐",
                font_size=45,
                falling_speed=5,
                animation_length="4"
            )
        except:
            pass

        st.warning("🟡 Moderate Mental Tiredness")

        st.write("""
### 😐 Recommendation
- Take a 10–15 minute break
- Reduce screen time
- Drink enough water
- Stretch and relax
- Avoid continuous work
""")

    elif prediction <= 80:

        try:
            rain(
                emoji="🔥",
                font_size=50,
                falling_speed=7,
                animation_length="5"
            )
        except:
            pass

        st.error("🟠 High Mental Tiredness")

        st.write("""
### 😓 Recommendation
- Your stress level is high
- Take adequate rest
- Reduce workload
- Sleep at least 7–8 hours
- Avoid excessive caffeine
""")

    else:

        try:
            rain(
                emoji="🔥",
                font_size=55,
                falling_speed=8,
                animation_length="6"
            )
        except:
            pass

        st.error("🔴 Very High Mental Tiredness")

        st.write("""
### 🚨 Immediate Recommendation
- Stop working immediately
- Take proper rest
- Sleep well
- Reduce workload
- Practice meditation
- Consult a healthcare professional if this persists
""")

    st.markdown("---")

    result = pd.DataFrame({
        "Prediction Score": [round(prediction, 2)]
    })

    st.download_button(
        "📥 Download Prediction",
        result.to_csv(index=False),
        "prediction.csv",
        "text/csv"
    )