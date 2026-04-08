"""
Emotion_Detection_System Dashboard
==================
Streamlit dashboard for the Emotion_Detection_System emotion detection system.
"""

import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime
import os
import time

st.set_page_config(page_title="Emotion_Detection_System Dashboard", page_icon="🧠", layout="wide")

DB_PATH = "Emotion_Detection_System_log.db"
NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card { background: #1c1f26; border: 1px solid #2e3340; border-radius: 12px; padding: 20px 24px; margin-bottom: 12px; }
    .metric-label { font-size: 0.75rem; color: #8891a4; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00ffb2; }
    .metric-value.alert { color: #ff4b4b; }
    .alert-banner { background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44); border: 1px solid #ff4b4b; border-radius: 10px; padding: 14px 20px; color: #ff4b4b; font-weight: 600; margin-bottom: 16px; }
    .section-header { font-size: 0.8rem; color: #8891a4; text-transform: uppercase; letter-spacing: 0.12em; border-bottom: 1px solid #2e3340; padding-bottom: 8px; margin: 20px 0 12px 0; }
</style>
""", unsafe_allow_html=True)

def load_recent(limit: int = 200) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["id", "timestamp", "emotion", "confidence"])
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        df = pd.read_sql_query(f"SELECT * FROM detections ORDER BY id DESC LIMIT {limit}", conn)
        conn.close()
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        df["emotion"] = df["emotion"].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x))
        df = df.dropna(subset=["confidence"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        return df
    except Exception:
        return pd.DataFrame(columns=["id", "timestamp", "emotion", "confidence"])

with st.sidebar:
    st.markdown("## Alert Configuration")
    alert_threshold = st.slider("Negative detections to trigger alert", 1, 20, 5)
    alert_window = st.slider("Time window (seconds)", 5, 120, 10, step=5)
    chart_limit = st.slider("Recent detections to show in chart", 20, 200, 60, step=10)
    st.markdown("---")
    st.markdown("## Info")
    st.markdown(f"**DB:** `{DB_PATH}`")
    st.markdown("**Refresh:** every 2 seconds")
    st.markdown("**Run app:** `python app.py`")

st.markdown("# Emotion_Detection_System")
st.markdown("<p style='color:#8891a4;margin-top:-12px;'>Real-time Emotion Detection Dashboard</p>", unsafe_allow_html=True)

df = load_recent(limit=chart_limit)

if df.empty:
    st.info("No detections yet. Make sure `app.py` is running and a face is visible.")
    time.sleep(2)
    st.rerun()

recent_window = df[df["timestamp"] >= (pd.Timestamp.now() - pd.Timedelta(seconds=alert_window))]
negative_in_window = recent_window[recent_window["emotion"].str.lower().isin(NEGATIVE_EMOTIONS)]
alert_firing = len(negative_in_window) >= alert_threshold

if alert_firing:
    st.markdown(f"<div class='alert-banner'>!! ALERT — {len(negative_in_window)} negative detections in the last {alert_window}s (threshold: {alert_threshold})</div>", unsafe_allow_html=True)

latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Current Emotion</div><div class='metric-value {'alert' if latest['emotion'].lower() in NEGATIVE_EMOTIONS else ''}'>{latest['emotion'].upper()}</div></div>", unsafe_allow_html=True)

with col2:
    try:
        conf_pct = f"{float(latest['confidence']) * 100:.1f}%"
    except:
        conf_pct = "N/A"
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Confidence</div><div class='metric-value'>{conf_pct}</div></div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Detections</div><div class='metric-value'>{len(df)}</div></div>", unsafe_allow_html=True)

with col4:
    neg_pct = f"{(df['emotion'].str.lower().isin(NEGATIVE_EMOTIONS).mean() * 100):.1f}%"
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Negative Rate</div><div class='metric-value'>{neg_pct}</div></div>", unsafe_allow_html=True)

emotion_colors = {"happy": "#00ffb2", "neutral": "#8891a4", "surprise": "#ffd700", "sad": "#4b9fff", "angry": "#ff4b4b", "fear": "#ff944b", "disgust": "#c44bff"}

st.markdown("<div class='section-header'>Emotion Timeline</div>", unsafe_allow_html=True)
fig = px.scatter(df, x="timestamp", y="confidence", color="emotion", color_discrete_map=emotion_colors, template="plotly_dark", height=320)
fig.update_traces(marker=dict(size=8, opacity=0.85))
fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1c1f26", legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("<div class='section-header'>Emotion Distribution</div>", unsafe_allow_html=True)
    dist = df["emotion"].value_counts().reset_index()
    dist.columns = ["emotion", "count"]
    fig2 = px.bar(dist, x="emotion", y="count", color="emotion", color_discrete_map=emotion_colors, template="plotly_dark", height=280)
    fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1c1f26", showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown("<div class='section-header'>Recent Detections</div>", unsafe_allow_html=True)
    display_df = df[["timestamp", "emotion", "confidence"]].tail(15).iloc[::-1].copy()
    display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M:%S")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}  •  Auto-refreshes every 2s")
time.sleep(2)
st.rerun()