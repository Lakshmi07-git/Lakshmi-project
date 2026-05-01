import streamlit as st
import cv2
import time
import numpy as np
import winsound
from datetime import datetime

st.set_page_config(page_title="AI-Proctor: Advanced Exam Monitor", layout="wide")

# Load OpenCV Classifiers (Python 3.13 compatible)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --- SESSION STATE INITIALIZATION ---
if 'phase' not in st.session_state:
    st.session_state.update({
        'phase': "login", 
        'violations': 0, 
        'last_viol_time': 0,
        'logs': [],
        'start_time': None
    })

# --- HELPER FUNCTIONS ---
def add_log(reason):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{timestamp}] {reason}")

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return "Low Lighting" if avg_brightness < 50 else "Good"

# --- PHASE 1: PROFESSIONAL LOGIN ---
if st.session_state.phase == "login":
    st.title("🎓 University Online Examination Portal")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Student Full Name")
        roll = st.text_input("Roll Number / Hall Ticket")
    with col2:
        subject = st.selectbox("Select Subject", ["Computer Vision", "AI & ML", "Database Systems"])
        exam_id = st.text_input("Unique Exam ID", value="EXAM-2024-001")
    
    if st.button("Authenticate & Start Exam"):
        if name and roll:
            st.session_state.update({'name': name, 'roll': roll, 'phase': "exam", 'start_time': time.time()})
            st.rerun()
        else:
            st.error("Authentication Failed: Please fill all fields.")

# --- PHASE 2: ADVANCED MONITORING ---
else:
    # Sidebar: Professional Dashboard
    st.sidebar.title("🛡️ Proctor Dashboard")
    st.sidebar.subheader("Student Details")
    st.sidebar.write(f"**Name:** {st.session_state.name}")
    st.sidebar.write(f"**ID:** {st.session_state.roll}")
    
    st.sidebar.divider()
    
    # Live Metrics
    viol_metric = st.sidebar.metric("Total Violations", st.session_state.violations)
    light_status = st.sidebar.empty()
    
    st.sidebar.subheader("Violation Logs")
    log_area = st.sidebar.container(height=200)
    
    # Main Exam Interface
    st.title(f"Ongoing Exam: {st.session_state.name}")
    frame_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Environment Check
        light = check_lighting(frame)
        light_status.write(f"Environment: **{light}**")
        
        # 2. Advanced Detection Logic
        faces = face_cascade.detectMultiScale(gray, 1.1, 8)
        status = "Focused"
        
        if len(faces) == 0:
            status = "FACE NOT DETECTED"
        elif len(faces) > 1:
            status = "MULTIPLE PEOPLE DETECTED"
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                
                # If eyes aren't detected, they are likely looking away
                if len(eyes) < 2:
                    status = "DISTRACTED / LOOKING AWAY"
                
                # Draw professional bounding boxes
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) if status == "Focused" else (0, 0, 255), 2)

        # 3. Smart Violation Engine
        curr_time = time.time()
        if status != "Focused":
            if curr_time - st.session_state.last_viol_time > 4: # 4 second buffer
                st.session_state.violations += 1
                st.session_state.last_viol_time = curr_time
                add_log(status)
                winsound.Beep(800, 400) # Warning Tone
        
        # 4. Security Overlay
        current_dt = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cv2.putText(frame, f"LIVE: {current_dt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"ID: {st.session_state.roll}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if status != "Focused" else (0, 255, 0), 2)

        # 5. UI Rendering
        frame_placeholder.image(frame, channels="BGR")
        viol_metric.metric("Total Violations", st.session_state.violations)
        with log_area:
            for log in st.session_state.logs:
                st.caption(log)

    cap.release()