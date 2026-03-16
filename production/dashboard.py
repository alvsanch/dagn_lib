"""
dashboard.py — dagn_lib Streamlit dashboard

Responsibilities:
  - MQTT subscription → accumulate raw sensor data (ir, gsr, tmp, att, med)
  - Camera display via Windows Flask MJPEG server
  - Every 800ms: POST latest raw values to /analyze service
  - Display: VA plane, raw signal charts, metrics from service response

NO signal processing here — NeuroKit2, peak detection, rPPG, blinks, SpO2
are all handled by analizar_emocion_service.py.
"""

import streamlit as st
import requests
import json
import queue
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import time

# ─── Config ───────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="DAGN-lib Dashboard")


st.title("DAGN-lib Multimodal Emotional Dashboard")

WINDOWS_IP  = "172.26.96.1"
SERVICE_URL = "http://localhost:8000/analyze"
FLASK_URL   = f"http://{WINDOWS_IP}:5000"
MQTT_BROKER = "localhost"
MQTT_PORT   = 1883

st_autorefresh(interval=1000, key="refresh")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe(v, default=0.0):
    """Return float, replacing NaN/Inf/errors with default."""
    try:
        v = float(v)
        return v if np.isfinite(v) else default
    except Exception:
        return default

# ─── MQTT ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def setup_mqtt():
    q = queue.Queue()

    def on_connect(client, userdata, flags, rc, properties=None):
        client.subscribe("tesis/biomedidas")

    def on_message(client, userdata, msg):
        try:
            q.put(json.loads(msg.payload.decode()))
        except Exception:
            pass

    client = mqtt.Client(CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    return client, q

mqtt_client, data_queue = setup_mqtt()

# ─── Session state ────────────────────────────────────────────────────────────

_defaults = {
    "running":          False,
    "session_id":       None,
    "sensor_buffer":    [],
    "va_history":       [],
    "start_ts":         None,
    "mqtt_total":       0,
    "last_msg_ts":      None,
    "last_physio_ts":   None,   # timestamp of last physio row sent to service
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar: session control ─────────────────────────────────────────────────

with st.sidebar:
    st.header("Session Control")

    if not st.session_state.running:
        if st.button("START SESSION"):
            sid = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.session_id    = sid
            st.session_state.running       = True
            st.session_state.va_history    = []
            st.session_state.start_ts      = time.time()
            st.session_state.mqtt_total    = 0
            st.session_state.last_msg_ts   = None
            st.session_state.last_physio_ts = datetime.now()  # only fresh data
            mqtt_client.publish("tesis/control", "START")
            try:
                requests.get(f"{FLASK_URL}/record_start",
                             params={"session_id": sid}, timeout=2)
            except Exception:
                pass
    else:
        if st.button("STOP SESSION"):
            mqtt_client.publish("tesis/control", "STOP")
            try:
                requests.get(f"{FLASK_URL}/record_stop", timeout=2)
            except Exception:
                pass
            st.session_state.running = False

    st.markdown("---")
    st.caption(f"Session: `{st.session_state.session_id or '—'}`")
    st.caption(f"Service: `{SERVICE_URL}`")

# ─── Drain MQTT queue ────────────────────────────────────────────────────────

new_msgs = 0
while not data_queue.empty():
    d = data_queue.get()
    d["timestamp"] = datetime.now()
    st.session_state.sensor_buffer.append(d)
    if len(st.session_state.sensor_buffer) > 1000:
        st.session_state.sensor_buffer.pop(0)
    new_msgs += 1

if new_msgs > 0:
    st.session_state.mqtt_total += new_msgs
    st.session_state.last_msg_ts = time.time()

# ─── Session status banner ───────────────────────────────────────────────────

if st.session_state.running and st.session_state.start_ts is not None:
    elapsed = time.time() - st.session_state.start_ts
    msgs    = st.session_state.mqtt_total
    if msgs == 0:
        if elapsed < 10:
            st.info("Starting session... waiting for sensor data")
        else:
            st.warning(
                f"Session active ({int(elapsed)}s) but **no MQTT data received**. "
                "Check that sensors are on and publishing to `tesis/biomedidas`."
            )
    else:
        secs_since = time.time() - st.session_state.last_msg_ts if st.session_state.last_msg_ts else 999
        if secs_since < 3:
            st.success(f"Receiving data — {msgs} messages received")
        else:
            st.warning(f"Last data {int(secs_since)}s ago — {msgs} total messages")

# ─── Build DataFrame from sensor buffer ──────────────────────────────────────

df = None
if st.session_state.sensor_buffer:
    df = pd.DataFrame(st.session_state.sensor_buffer).sort_values("timestamp")

# ─── Extract latest raw sensor values (NO processing) ────────────────────────

ir_val = red_val = gsr_val = temp_val = att_val = med_val = 0.0
ir_batch = red_batch = gsr_batch = temp_batch = []

if df is not None:
    # Physio sensor: ir, red, gsr, tmp (rows with ir present)
    if "ir" in df.columns and df["ir"].notna().any():
        physio_df = df[df["ir"].notna()].copy()

        # Collect new samples since last call (for true 50 Hz HRV/SpO2)
        last_ts = st.session_state.last_physio_ts
        new_physio = (
            physio_df[physio_df["timestamp"] > last_ts]
            if last_ts is not None
            else pd.DataFrame()   # last_physio_ts always set on session start
        )
        if len(new_physio) > 0:
            ir_batch   = [safe(v) for v in new_physio["ir"].tolist()]
            red_batch  = [safe(v) for v in new_physio["red"].tolist()] if "red" in new_physio else [0.0] * len(new_physio)
            gsr_batch  = [safe(v) for v in new_physio["gsr"].tolist()] if "gsr" in new_physio else [0.0] * len(new_physio)
            temp_batch = [safe(v) for v in new_physio["tmp"].tolist()] if "tmp" in new_physio else [0.0] * len(new_physio)
            st.session_state.last_physio_ts = new_physio["timestamp"].iloc[-1]

        # Latest single values for display
        last_physio = physio_df.iloc[-1]
        ir_val   = safe(last_physio.get("ir",  0.0))
        red_val  = safe(last_physio.get("red", 0.0))
        gsr_val  = safe(last_physio.get("gsr", 0.0))
        temp_val = safe(last_physio.get("tmp", 0.0))

    # EEG sensor: att, med (rows with att present)
    if "att" in df.columns and df["att"].notna().any():
        last_eeg = df[df["att"].notna()].iloc[-1]
        att_val = safe(last_eeg.get("att", 0.0))
        med_val = safe(last_eeg.get("med", 0.0))

# ─── Call inference service ──────────────────────────────────────────────────

svc_data = {"status": "idle"}

if st.session_state.running and st.session_state.session_id:
    payload = {
        "session_id": st.session_state.session_id,
        "ir_batch":   ir_batch,
        "red_batch":  red_batch,
        "gsr_batch":  gsr_batch,
        "temp_batch": temp_batch,
        "ir":   ir_val,
        "red":  red_val,
        "gsr":  gsr_val,
        "temp": temp_val,
        "att":  att_val,
        "med":  med_val,
    }
    try:
        r = requests.post(SERVICE_URL, json=payload, timeout=3)
        r.raise_for_status()
        svc_data = r.json()
    except requests.exceptions.Timeout:
        svc_data = {"status": "error", "detail": "timeout"}
    except requests.exceptions.ConnectionError:
        svc_data = {"status": "error", "detail": "service unavailable"}
    except Exception as e:
        svc_data = {"status": "error", "detail": str(e)}

    if svc_data.get("status") == "success":
        st.session_state.va_history.append({
            "time":    datetime.now(),
            "valence": safe(svc_data.get("valence", 0.0)),
            "arousal": safe(svc_data.get("arousal", 0.0)),
            "grad_v":  safe(svc_data.get("grad_v", 0.0)),
            "grad_a":  safe(svc_data.get("grad_a", 0.0)),
            "face_v":  safe(svc_data.get("face_v", 0.0)),
            "face_a":  safe(svc_data.get("face_a", 0.0)),
        })

# ─── Layout principal: contenido (izquierda) | métricas (derecha) ─────────────

col_main, col_metrics = st.columns([4, 1])

with col_metrics:
    st.subheader("Metrics")

    hr_bpm     = safe(svc_data.get("hr_bpm",     0.0))
    rmssd      = safe(svc_data.get("rmssd",      0.0))
    eda_mean   = safe(svc_data.get("eda_mean",   0.0))
    spo2       = safe(svc_data.get("spo2",       0.0))
    cam_hr     = safe(svc_data.get("cam_hr",     0.0))
    cam_resp   = safe(svc_data.get("cam_resp",   0.0))
    blink_rate = safe(svc_data.get("blink_rate", 0.0))

    st.caption("**Sensor**")
    st.metric("HR sensor (BPM)", f"{hr_bpm:.1f}" if hr_bpm > 0 else "—")
    st.metric("RMSSD (ms)",      f"{rmssd:.1f}"  if rmssd  > 0 else "—")
    st.metric("EDA tonic",       f"{eda_mean:.3f}")
    st.metric("SpO2 (%)",        f"{spo2:.1f}"   if spo2   > 0 else "—")
    st.metric("Temp (°C)",       f"{temp_val:.1f}")
    st.metric("Attention",       f"{att_val:.0f}")
    st.metric("Meditation",      f"{med_val:.0f}")

    st.caption("**Camera**")
    st.metric("HR cam (BPM)",    f"{cam_hr:.1f}"     if cam_hr     > 0 else "—")
    st.metric("Resp (br/min)",   f"{cam_resp:.1f}"   if cam_resp   > 0 else "—")
    st.metric("Blinks (br/min)", f"{blink_rate:.1f}" if blink_rate > 0 else "—")

with col_main:
    # ── Fila superior: Cámara | Plano emocional ──────────────────────────────
    col_cam, col_va = st.columns(2)

    with col_cam:
        st.subheader("Camera")
        st.image(f"{FLASK_URL}/video_feed", width=320)

    with col_va:
        st.subheader("Emotional Plane")

        status = svc_data.get("status", "idle")
        if status == "warming_up":
            warmup_pct = safe(svc_data.get("warmup", 0.0)) * 100
            st.progress(int(warmup_pct), text=f"Warming up... {warmup_pct:.0f}%")
        elif status == "error":
            st.error(f"Service error: {svc_data.get('detail', 'unknown')}")

        if st.session_state.va_history:
            last   = st.session_state.va_history[-1]
            v, a   = last["valence"], last["arousal"]
            v_proj = v + last["grad_v"]
            a_proj = a + last["grad_a"]
            fv, fa = last.get("face_v", 0.0), last.get("face_a", 0.0)

            fig_va = go.Figure()
            fig_va.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0,
                             line=dict(color="gray", width=1))
            fig_va.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1,
                             line=dict(color="gray", width=1))
            for (tx, ty, label) in [(0.5, 0.8, "Happy"), (-0.5, 0.8, "Angry"),
                                     (0.5, -0.8, "Calm"), (-0.5, -0.8, "Sad")]:
                fig_va.add_annotation(x=tx, y=ty, text=label, showarrow=False,
                                      font=dict(color="lightgray", size=10))
            fig_va.add_trace(go.Scatter(
                x=[fv], y=[fa], mode="markers+text",
                text=["Cam"], textposition="bottom left",
                marker=dict(size=10, color="orange", symbol="diamond"),
                name="Camera only",
            ))
            fig_va.add_trace(go.Scatter(
                x=[v], y=[a], mode="markers+text",
                text=["Now"], textposition="top right",
                marker=dict(size=16, color="royalblue"),
                name="Fusion",
            ))
            fig_va.add_trace(go.Scatter(
                x=[v_proj], y=[a_proj], mode="markers",
                marker=dict(size=10, color="red", symbol="x"),
                name="Projected",
            ))
            fig_va.add_trace(go.Scatter(
                x=[v, v_proj], y=[a, a_proj], mode="lines",
                line=dict(color="red", dash="dot"),
                showlegend=False,
            ))
            fig_va.update_layout(
                xaxis=dict(range=[-1, 1], title="Valence", zeroline=False),
                yaxis=dict(range=[-1, 1], title="Arousal", zeroline=False),
                height=340, showlegend=True,
                legend=dict(orientation="h", y=-0.25, x=0, yanchor="top"),
                margin=dict(l=30, r=10, t=10, b=70),
            )
            st.plotly_chart(fig_va, use_container_width=True)
            if status == "success":
                st.caption(
                    f"Fusion — V: **{v:.3f}**  A: **{a:.3f}** &nbsp;&nbsp;|&nbsp;&nbsp;"
                    f"Camera — V: **{fv:.3f}**  A: **{fa:.3f}**"
                )
        else:
            st.caption("No VA predictions yet — start a session and wait for warmup")

    # ── VA Timeline ───────────────────────────────────────────────────────────
    if len(st.session_state.va_history) > 3:
        st.subheader("Valence & Arousal Timeline")
        df_va = pd.DataFrame(st.session_state.va_history)
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=df_va["time"], y=df_va["valence"],
                                   mode="lines", name="Valence",
                                   line=dict(color="royalblue", width=2),
                                   connectgaps=True))
        fig_t.add_trace(go.Scatter(x=df_va["time"], y=df_va["arousal"],
                                   mode="lines", name="Arousal",
                                   line=dict(color="tomato", width=2),
                                   connectgaps=True))
        fig_t.update_layout(
            yaxis=dict(range=[-1.05, 1.05]),
            height=180,
            margin=dict(l=30, r=10, t=10, b=30),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig_t, use_container_width=True, key="va_timeline")

    # ── Raw signal charts ─────────────────────────────────────────────────────
    if df is not None:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Raw IR (BVP proxy)")
            if "ir" in df.columns and df["ir"].notna().any():
                df_ir = df[["timestamp", "ir"]].dropna(subset=["ir"])
                fig_ir = go.Figure(go.Scatter(
                    x=df_ir["timestamp"], y=df_ir["ir"],
                    mode="lines", name="IR", line=dict(color="steelblue"),
                ))
                fig_ir.update_layout(
                    height=180, margin=dict(l=0, r=0, t=10, b=30),
                    xaxis_title="", yaxis_title="Raw IR",
                )
                st.plotly_chart(fig_ir, use_container_width=True)
            else:
                st.caption("No IR data yet")

        with c2:
            st.subheader("GSR Signal")
            if "gsr" in df.columns and df["gsr"].notna().any():
                df_gsr = df[["timestamp", "gsr"]].dropna(subset=["gsr"])
                fig_gsr = go.Figure(go.Scatter(
                    x=df_gsr["timestamp"], y=df_gsr["gsr"],
                    mode="lines", name="GSR", line=dict(color="seagreen"),
                ))
                fig_gsr.update_layout(
                    height=180, margin=dict(l=0, r=0, t=10, b=30),
                    xaxis_title="", yaxis_title="GSR (μS)",
                )
                st.plotly_chart(fig_gsr, use_container_width=True)
            else:
                st.caption("No GSR data yet")

        with c3:
            st.subheader("Attention / Meditation")
            has_att = "att" in df.columns and df["att"].notna().any()
            has_med = "med" in df.columns and df["med"].notna().any()
            if has_att or has_med:
                fig_eeg = go.Figure()
                if has_att:
                    df_att = df[["timestamp", "att"]].dropna(subset=["att"])
                    fig_eeg.add_trace(go.Scatter(
                        x=df_att["timestamp"], y=df_att["att"],
                        mode="lines", name="Attention", line=dict(color="darkorchid"),
                    ))
                if has_med:
                    df_med = df[["timestamp", "med"]].dropna(subset=["med"])
                    fig_eeg.add_trace(go.Scatter(
                        x=df_med["timestamp"], y=df_med["med"],
                        mode="lines", name="Meditation", line=dict(color="darkorange"),
                    ))
                fig_eeg.update_layout(
                    yaxis=dict(range=[0, 100]),
                    height=180, margin=dict(l=0, r=0, t=10, b=30),
                    xaxis_title="", yaxis_title="NeuroSky (0-100)",
                )
                st.plotly_chart(fig_eeg, use_container_width=True)
            else:
                st.caption("No EEG data yet")
