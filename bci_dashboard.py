import streamlit as st
import pandas as pd
import numpy as np
import time
import random

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Neuro-Swarm BCI Copilot", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- 2. Custom Neon CSS ---
# This matches the purple/cyan cyberpunk aesthetic of your poster
st.markdown("""
<style>
    .action-text { font-size: 55px !important; color: #00FFCC; font-weight: 900; text-align: center; text-shadow: 0px 0px 10px #00FFCC;}
    .resting-text { font-size: 55px !important; color: #444444; font-weight: 900; text-align: center;}
    .status-text { font-size: 22px !important; color: #CC99FF; text-align: center;}
    .metric-label { font-size: 18px !important; color: #FFFFFF;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Neuro-Swarm: Live BCI Decoding Copilot")
st.markdown("---")

# --- 3. Dashboard Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live EEG Feed (8-Channel Array)")
    chart_placeholder = st.empty()

with col2:
    st.subheader("AI Decoding Status")
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    status_placeholder = st.empty()
    action_placeholder = st.empty()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Neural Decoding Confidence:</p>', unsafe_allow_html=True)
    confidence_placeholder = st.empty()

# --- 4. Simulation Loop ---
channels = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
actions = ["RESTING", "GRASP (Prosthetic)", "PINCH (Precision)", "SWARM DEPLOY"]

# This loop runs endlessly to update the dashboard
while True:
    # Generate 50 data points of noisy sine waves for 8 channels
    x = np.linspace(0, 10, 50)
    data = {ch: np.sin(x + random.uniform(0, 5)) + np.random.normal(0, 0.4, 50) for ch in channels}
    df = pd.DataFrame(data)

    # Update the live chart
    chart_placeholder.line_chart(df, height=450, color=["#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#00FF00", "#0000FF", "#FFFFFF", "#FF8800"])

    # Simulate the AI Model classifying the brainwaves
    if random.random() > 0.85: # 15% chance to trigger an action
        action = random.choice(actions[1:])
        conf = random.uniform(88.5, 99.9)
        
        status_placeholder.markdown('<p class="status-text">Status: <b>MOTOR INTENT DETECTED</b></p>', unsafe_allow_html=True)
        action_placeholder.markdown(f'<p class="action-text">{action}</p>', unsafe_allow_html=True)
        confidence_placeholder.progress(int(conf), text=f"{conf:.1f}%")
        
    else:
        status_placeholder.markdown('<p class="status-text">Status: Scanning Neural Patterns...</p>', unsafe_allow_html=True)
        action_placeholder.markdown('<p class="resting-text">RESTING</p>', unsafe_allow_html=True)
        confidence_placeholder.progress(0, text="0.0%")

    # Pause briefly before the next data frame
    time.sleep(0.4)