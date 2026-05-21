"""
app.py  —  Sign Language to Text   (MediaPipe landmark edition)
"""

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque, Counter
from hand_detection import extract_landmarks, draw_bboxes

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:       #0a0a0f;
    --surface:  #12121a;
    --border:   #1e1e2e;
    --accent:   #00ff9d;
    --accent2:  #7c3aed;
    --text:     #e2e8f0;
    --muted:    #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Header */
.hero {
    text-align: center;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent), #00c6ff, var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero p {
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin-top: 4px;
}

/* Prediction card */
.pred-card {
    background: linear-gradient(135deg, #0f1a1f, #0a1628);
    border: 1px solid var(--accent);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,255,157,0.12);
    margin-bottom: 1rem;
}
.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
}
.pred-value {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    margin: 8px 0;
    text-shadow: 0 0 20px rgba(0,255,157,0.4);
}
.pred-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #00c6ff;
}

/* Sentence builder */
.sentence-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: var(--accent);
    letter-spacing: 2px;
    min-height: 56px;
    word-break: break-all;
}
.sentence-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
}

/* Stats row */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}
.stat-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
}
.stat-label {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Confidence bar */
.conf-bar-bg {
    background: var(--border);
    border-radius: 6px;
    height: 8px;
    margin-top: 4px;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    transition: width 0.3s ease;
}

/* History chips */
.history-chip {
    display: inline-block;
    background: var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    margin: 2px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--text);
}

/* Streamlit widget overrides */
.stButton > button {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: var(--accent);
    color: #000;
}

div[data-testid="stCheckbox"] label {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.stSlider > div > div { background: var(--border) !important; }
.stSlider [data-testid="stSlider"] div div div div {
    background: var(--accent) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────
if "sentence"      not in st.session_state: st.session_state.sentence      = ""
if "history"       not in st.session_state: st.session_state.history       = []
if "frame_count"   not in st.session_state: st.session_state.frame_count   = 0
if "detect_count"  not in st.session_state: st.session_state.detect_count  = 0
if "last_added"    not in st.session_state: st.session_state.last_added     = ""
if "fps_buffer"    not in st.session_state: st.session_state.fps_buffer    = deque(maxlen=20)
if "vote_buffer"   not in st.session_state: st.session_state.vote_buffer   = deque(maxlen=15)

# ── Hero header ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🤟 Sign Language conversion</h1>
  <p>MediaPipe Landmark Engine · Real-time · High Accuracy</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    run = st.checkbox("▶ Start Camera", value=False)
    st.divider()

    st.markdown("**Prediction Settings**")
    conf_threshold = st.slider("Min Confidence", 0.3, 1.0, 0.6, 0.05,
                                help="Ignore predictions below this confidence")
    vote_window    = st.slider("Smoothing Window", 5, 30, 15,
                                help="Frames to average for stable prediction")
    add_delay      = st.slider("Letter Add Delay (s)", 0.5, 3.0, 1.5, 0.5,
                                help="Wait this long before adding a letter to sentence")
    st.divider()

    st.markdown("**Sentence Builder**")
    if st.button("⌫  Backspace"):
        if st.session_state.sentence:
            st.session_state.sentence = st.session_state.sentence[:-1]
    if st.button("▭  Space"):
        st.session_state.sentence += " "
    if st.button("🗑  Clear Sentence"):
        st.session_state.sentence = ""
        st.session_state.history  = []
    st.divider()
    
    
    

    st.markdown("**Model**")
    from predict import model_ready
    if model_ready():
        st.success("Model loaded ✅")
        from predict import predict_sign
    else:
        st.error("No model found!\nRun: `python train.py`")
        predict_sign = None

    st.divider()
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#64748b;line-height:1.8'>
    <b style='color:#e2e8f0'>Quick Start</b><br>
    1. Collect data per gesture:<br>
    &nbsp;&nbsp;<code>python collect_data.py --label A</code><br>
    2. Train the model:<br>
    &nbsp;&nbsp;<code>python train.py</code><br>
    3. Start this app ✅
    </div>
    """, unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────
col_cam, col_info = st.columns([3, 2], gap="large")

with col_cam:
    cam_placeholder = st.empty()

with col_info:
    pred_placeholder     = st.empty()
    conf_bar_placeholder = st.empty()
    sentence_placeholder = st.empty()
    stats_placeholder    = st.empty()
    history_placeholder  = st.empty()

# ── Initial render (idle state) ───────────────────────────────────
def render_idle():
    with cam_placeholder:
        st.markdown("""
        <div style="height:400px;background:#12121a;border:1px dashed #1e1e2e;
                    border-radius:16px;display:flex;align-items:center;
                    justify-content:center;flex-direction:column;gap:12px">
          <div style="font-size:3rem">📷</div>
          <div style="font-family:Space Mono,monospace;font-size:0.8rem;color:#64748b">
            Enable camera in the sidebar to begin
          </div>
        </div>""", unsafe_allow_html=True)

    with pred_placeholder:
        st.markdown("""
        <div class="pred-card">
          <div class="pred-label">Current Sign</div>
          <div class="pred-value" style="color:#1e1e2e">—</div>
          <div class="pred-conf">waiting for camera...</div>
        </div>""", unsafe_allow_html=True)

    with sentence_placeholder:
        st.markdown('<div class="sentence-label">Sentence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sentence-box">&nbsp;</div>', unsafe_allow_html=True)

render_idle()

# ── Camera loop ───────────────────────────────────────────────────
if run and predict_sign is not None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_letter_time = time.time()
    prev_time        = time.time()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera not accessible")
            break

        frame = cv2.flip(frame, 1)
        st.session_state.frame_count += 1

        # FPS
        now      = time.time()
        elapsed  = now - prev_time
        prev_time = now
        fps       = 1.0 / elapsed if elapsed > 0 else 0
        st.session_state.fps_buffer.append(fps)
        avg_fps  = np.mean(st.session_state.fps_buffer)

        # Extract landmarks
        features, annotated, bboxes = extract_landmarks(frame)

        label, confidence = "—", 0.0
        stable_pred       = None

        if features is not None and np.any(features != 0):
            st.session_state.detect_count += 1
            label, confidence = predict_sign(features)

            # Voting buffer for stable prediction
            if confidence >= conf_threshold:
                st.session_state.vote_buffer.append(label)

            if len(st.session_state.vote_buffer) >= vote_window:
                counter      = Counter(st.session_state.vote_buffer)
                top, count   = counter.most_common(1)[0]
                vote_ratio   = count / len(st.session_state.vote_buffer)
                if vote_ratio >= 0.6:
                    stable_pred = top

                    # Auto-add to sentence after delay
                    if (stable_pred != st.session_state.last_added and
                            now - last_letter_time >= add_delay):
                        st.session_state.sentence   += stable_pred
                        st.session_state.history.append(stable_pred)
                        st.session_state.last_added  = stable_pred
                        last_letter_time             = now

            # Overlay on frame
            annotated = draw_bboxes(annotated, bboxes, label if confidence >= conf_threshold else None, confidence)

        # Dark overlay bars for HUD
        hud_h = 50
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (annotated.shape[1], hud_h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        cv2.putText(annotated, f"FPS: {avg_fps:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,157), 2)
        cv2.putText(annotated, f"SIGN: {label}  {confidence:.0%}" if label != "—" else "NO HAND",
                    (annotated.shape[1]//2 - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,157) if confidence >= conf_threshold else (100,100,100), 2)

        # ── Update UI ─────────────────────────────────────────────
        cam_placeholder.image(annotated, channels="BGR", width=640)

        # Prediction card
        display_label = stable_pred if stable_pred else (label if confidence >= conf_threshold else "—")
        bar_pct       = int(confidence * 100)
        bar_color     = "#00ff9d" if confidence >= conf_threshold else "#64748b"

        pred_placeholder.markdown(f"""
        <div class="pred-card">
          <div class="pred-label">Current Sign</div>
          <div class="pred-value">{display_label}</div>
          <div class="pred-conf">confidence: {confidence:.0%}</div>
        </div>""", unsafe_allow_html=True)

        conf_bar_placeholder.markdown(f"""
        <div style="margin-bottom:0.8rem">
          <div style="display:flex;justify-content:space-between;
                      font-family:Space Mono,monospace;font-size:0.65rem;color:#64748b;margin-bottom:4px">
            <span>CONFIDENCE</span><span>{bar_pct}%</span>
          </div>
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{bar_pct}%;background:{'linear-gradient(90deg,#7c3aed,#00ff9d)' if confidence>=conf_threshold else '#1e1e2e'}"></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Sentence
        sentence_placeholder.markdown(f"""
        <div class="sentence-label">Built Sentence</div>
        <div class="sentence-box">{st.session_state.sentence or '&nbsp;'}</div>
        """, unsafe_allow_html=True)

        # Stats
        det_rate = (st.session_state.detect_count / st.session_state.frame_count * 100
                    if st.session_state.frame_count else 0)
        stats_placeholder.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin:0.8rem 0">
          <div class="stat-card"><div class="stat-num">{avg_fps:.0f}</div><div class="stat-label">FPS</div></div>
          <div class="stat-card"><div class="stat-num">{det_rate:.0f}%</div><div class="stat-label">Detect</div></div>
          <div class="stat-card"><div class="stat-num">{len(st.session_state.sentence)}</div><div class="stat-label">Letters</div></div>
        </div>""", unsafe_allow_html=True)

        # History chips
        recent = st.session_state.history[-20:]
        chips  = "".join(f'<span class="history-chip">{c}</span>' for c in recent)
        history_placeholder.markdown(f"""
        <div style="margin-top:0.4rem">
          <div class="sentence-label">History</div>
          <div style="margin-top:6px">{chips or '<span style="color:#64748b;font-size:0.8rem;font-family:Space Mono">—</span>'}</div>
        </div>""", unsafe_allow_html=True)

    cap.release()

elif run and predict_sign is None:
    st.warning("⚠️ Please train the model first before starting the camera.")
    render_idle()
