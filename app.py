"""
Streamlit dashboard for the security_pipeline project.

Run with (from the security_pipeline/ directory):
    streamlit run app.py

Uses my_camera.json if present in this directory, otherwise the built-in
defaults from config/camera_config.py. All detection logic still lives in
core/, unchanged - engine.py just adapts it to a background thread and
shared state that this file polls and renders. See DASHBOARD_README.md
for setup, deployment, and how the zone editor / settings tabs work.

Visual theme lives in .streamlit/config.toml (native widgets: buttons,
sliders, tabs, inputs) - the <style> block below only covers the bespoke
elements theming can't reach (status bar, video frame, alert cards,
section headers).

Layout:
  sidebar   - Start/Stop, hazard-check frequency, detection zone + ignore
              zone editors (these need the live feed for their preview
              overlay, so they stay next to it rather than in a tab).
  main area - the live feed + alerts, then three tabs for everything else
              main.py's CLI already supported but the dashboard didn't
              yet expose: camera source, both schedules, sensitivity and
              detection-threshold tuning, and known-face enrollment.
"""

import base64
import html
import json
import time
import uuid
from datetime import time as dtime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from config.camera_config import load_camera_config, DEFAULT_CONFIG, SENSITIVITY_PRESETS
from core.schedule import get_current_mode, ALL_DAYS
from engine import SecurityEngine, SharedState

CONFIG_PATH = "my_camera.json"

st.set_page_config(page_title="Security Monitor", page_icon="🔴", layout="wide")

# ---------------------------------------------------------------- styling --
# Colors/radii/fonts for native widgets (buttons, sliders, tabs, inputs,
# checkboxes) come from .streamlit/config.toml, not from here - that's the
# Streamlit-recommended way to theme them, and it won't break on a
# version upgrade the way selector-based CSS can. This block covers only
# the custom elements below that theming doesn't reach.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700;700&display=swap');

:root {
    --amber: #e8a33d; --red: #ef5555; --green: #3ecf8e; --violet: #a78bfa;
    --primary: var(--st-primary-color, #3ea6ff);
    --bg: var(--st-background-color, #0f1318);
    --panel: var(--st-secondary-background-color, #1b2127);
    --border: var(--st-border-color, #2a323a);
    --text: var(--st-text-color, #e4e9ef);
    --text-dim: #7a8492;
}
h1, h2, h3, h4, h5, .app-header, .section-header { font-family: 'Space Grotesk', sans-serif; }
.mono { font-family: 'IBM Plex Mono', monospace; }

.app-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.1rem; }
.app-header .dot { width: 11px; height: 11px; border-radius: 50%;
    background: var(--primary); box-shadow: 0 0 10px var(--primary); flex-shrink: 0; }
.app-header h3 { margin: 0; font-size: 1.3rem; font-weight: 700;
    background: linear-gradient(135deg, var(--text) 40%, var(--primary));
    -webkit-background-clip: text; background-clip: text; color: transparent; }

.status-bar {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.7rem 1.1rem; margin-bottom: 1rem;
    background: linear-gradient(135deg, var(--panel), var(--bg) 140%);
    border: 1px solid var(--border); border-radius: 10px;
}
.status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.status-dot.live { background: var(--green); box-shadow: 0 0 10px var(--green); }
.status-dot.stopped { background: var(--text-dim); }
.rec-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--red);
    display: inline-block; animation: pulse 1.2s infinite; box-shadow: 0 0 8px var(--red); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

.video-frame {
    border: 1px solid var(--border); border-radius: 10px; padding: 6px;
    background: #000; position: relative;
}
.video-frame::before, .video-frame::after {
    content: ""; position: absolute; width: 18px; height: 18px;
    border-color: var(--primary); border-style: solid; opacity: 0.85;
}
.video-frame::before { top: 4px; left: 4px; border-width: 2px 0 0 2px; }
.video-frame::after { bottom: 4px; right: 4px; border-width: 0 2px 2px 0; }

.alert-card { border-left: 3px solid var(--text-dim); background: var(--panel);
    border-radius: 6px; padding: 0.65rem 0.9rem; margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.28); }
.alert-card.hazard { border-left-color: var(--red); }
.alert-card.unauthorized { border-left-color: var(--amber); }
.alert-card.safe { border-left-color: var(--green); }
.alert-title { font-weight: 600; font-size: 0.95rem; }
.alert-meta { color: var(--text-dim); font-size: 0.78rem; font-family: 'IBM Plex Mono', monospace; }

.section-caption { color: var(--text-dim); font-size: 0.85rem; margin-bottom: 0.6rem; }

.section-header { display: flex; align-items: center; gap: 0.55rem;
    font-weight: 600; font-size: 1.03rem; margin-bottom: 0.5rem; }
.section-header .bar { width: 4px; height: 1.15rem; border-radius: 2px;
    background: var(--primary); display: inline-block; flex-shrink: 0; }

/* Semantic accents on specific buttons via key= (stable, version-safe -
   Streamlit generates .st-key-<key> around any widget given a key=). */
.st-key-start_btn button { background: var(--green) !important; border-color: var(--green) !important; color: #06120c !important; }
.st-key-stop_btn button { background: var(--red) !important; border-color: var(--red) !important; color: #1a0808 !important; }
.st-key-save_zone_btn button { border-color: var(--amber) !important; color: var(--amber) !important; }
.st-key-save_ignore_btn button { border-color: var(--violet) !important; color: var(--violet) !important; }
</style>
""", unsafe_allow_html=True)


def _section_header(icon: str, label: str):
    st.markdown(f'<div class="section-header"><span class="bar"></span>{icon} {label}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- config io --

def _load_cfg():
    if Path(CONFIG_PATH).exists():
        return load_camera_config(CONFIG_PATH)
    return dict(DEFAULT_CONFIG)


def _save_cfg(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _parse_hhmm(s: str) -> dtime:
    h, m = map(int, s.split(":"))
    return dtime(hour=h, minute=m)


def _fmt_hhmm(t: dtime) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"


# ------------------------------------------- shared engine (survives reruns/refreshes) --

@st.cache_resource
def get_engine():
    return SecurityEngine(_load_cfg(), SharedState())


engine = get_engine()
state = engine.state


# ---------------------------------------------------------------- alert rendering --

def _is_notable(alert):
    if alert.get("type") == "hazard":
        return True
    decision = alert.get("decision", {})
    if decision.get("mode") == "cctv_mode":
        return False
    return decision.get("final_action") in ("SAFE_ENTRY", "UNAUTHORIZED_ALERT_UNKNOWN_FACE")


def _render_alert(alert):
    ts = time.strftime("%H:%M:%S", time.localtime(alert.get("timestamp", time.time())))
    decision = alert.get("decision", {})

    if alert.get("type") == "hazard":
        detail = decision.get("detail") or {}
        css_class = "hazard"
        title = f"\U0001F525 {detail.get('class_name', 'hazard').upper()} detected"
        meta = f"confidence {detail.get('confidence', 0):.2f}"
    else:
        final_action = decision.get("final_action", "LOG_ONLY")
        if final_action == "SAFE_ENTRY":
            css_class, title = "safe", "\u2713 Authorized entry"
            name = (decision.get("identity") or {}).get("detail", {}).get("matched_name", "unknown")
            meta = f"matched: {html.escape(str(name))}"
        elif final_action == "UNAUTHORIZED_ALERT_UNKNOWN_FACE":
            css_class, title = "unauthorized", "\u26A0 Unrecognized face"
            meta = f"person score {decision.get('person_score', 0):.2f}"
        else:
            css_class, title = "", "Motion logged"
            meta = f"person score {decision.get('person_score', 0):.2f}"

    st.markdown(f"""
    <div class="alert-card {css_class}">
        <div class="alert-title">{title}</div>
        <div class="alert-meta">{ts} &middot; {meta}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar --

with st.sidebar:
    st.markdown(
        '<div class="app-header"><span class="dot"></span><h3>Security Monitor</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="mono" style="color:var(--text-dim)">source: {engine.cfg.get("source")}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    snap = state.snapshot()

    if not snap["running"]:
        if st.button("\u25B6 Start monitoring", type="primary", use_container_width=True, key="start_btn"):
            with st.spinner("Loading models — first start can take a minute..."):
                engine.load_models()
            engine.start()
            st.rerun()
    else:
        if st.button("\u25A0 Stop monitoring", use_container_width=True, key="stop_btn"):
            engine.stop()
            st.rerun()

    st.markdown("---")
    with st.container(border=True):
        _section_header("\U0001F525", "Hazard check frequency")
        hz_interval = st.slider(
            "Seconds between fire/smoke checks", 0.5, 15.0,
            value=float(engine.cfg.get("hazard_check_interval", 2.0)), step=0.5,
            label_visibility="collapsed",
        )
        if hz_interval != engine.cfg.get("hazard_check_interval", 2.0):
            engine.update_hazard_interval(hz_interval)
            _save_cfg(engine.cfg)
        st.caption("A check that takes longer than this to run is never overlapped by the next one.")

    with st.container(border=True):
        _section_header("\U0001F3AF", "Detection zone")
        st.caption("Only motion inside this box is watched, recorded, and scored.")

        ref_h, ref_w = snap["frame_shape"] if snap["frame_shape"] else (480, 640)
        existing = engine.cfg.get("detection_zones") or []
        if existing and existing[0]:
            xs = [p[0] for p in existing[0]]
            ys = [p[1] for p in existing[0]]
            default_rect = (min(xs), min(ys), max(xs), max(ys))
        else:
            default_rect = (0, 0, ref_w, ref_h)

        zx1 = st.slider("Left", 0, ref_w, min(default_rect[0], ref_w))
        zx2 = st.slider("Right", 0, ref_w, min(default_rect[2], ref_w))
        zy1 = st.slider("Top", 0, ref_h, min(default_rect[1], ref_h))
        zy2 = st.slider("Bottom", 0, ref_h, min(default_rect[3], ref_h))
        st.caption("Amber box on the feed previews this before you save.")

        if st.button("Save zone", use_container_width=True, key="save_zone_btn"):
            engine.update_detection_zone((zx1, zy1, zx2, zy2))
            _save_cfg(engine.cfg)
            st.success("Zone updated — applies immediately, no restart needed.")

        with st.expander("Ignore zone (optional)"):
            st.caption("Motion inside this box is never watched or recorded at all — "
                        "use it for a ceiling fan, a tree branch, anything that keeps false-triggering.")
            ignore_enabled = st.checkbox("Enable ignore zone", value=bool(engine.cfg.get("ignore_zones")))

            ix1 = ix2 = iy1 = iy2 = None
            if ignore_enabled:
                existing_ig = engine.cfg.get("ignore_zones") or []
                if existing_ig and existing_ig[0]:
                    ixs = [p[0] for p in existing_ig[0]]
                    iys = [p[1] for p in existing_ig[0]]
                    default_irect = (min(ixs), min(iys), max(ixs), max(iys))
                else:
                    default_irect = (0, 0, min(150, ref_w), min(150, ref_h))

                ix1 = st.slider("Left", 0, ref_w, min(default_irect[0], ref_w), key="ig_left")
                ix2 = st.slider("Right", 0, ref_w, min(default_irect[2], ref_w), key="ig_right")
                iy1 = st.slider("Top", 0, ref_h, min(default_irect[1], ref_h), key="ig_top")
                iy2 = st.slider("Bottom", 0, ref_h, min(default_irect[3], ref_h), key="ig_bottom")
                st.caption("Violet box on the feed previews this before you save.")

                if st.button("Save ignore zone", use_container_width=True, key="save_ignore_btn"):
                    engine.update_ignore_zone((ix1, iy1, ix2, iy2))
                    _save_cfg(engine.cfg)
                    st.success("Ignore zone saved.")
            elif engine.cfg.get("ignore_zones"):
                if st.button("Clear ignore zone", use_container_width=True):
                    engine.update_ignore_zone(None)
                    _save_cfg(engine.cfg)
                    st.success("Ignore zone cleared.")

    st.markdown("---")
    show_routine = st.checkbox("Show routine activity (not just alerts)", value=False)


# ---------------------------------------------------------------- live view --

@st.fragment(run_every=0.35)
def live_view():
    snap = state.snapshot()
    current_mode = get_current_mode(engine.cfg)

    dot_class = "live" if snap["running"] else "stopped"
    if snap["is_recording"]:
        rec_html = ('<span class="rec-dot"></span> '
                    '<span class="mono" style="color:var(--red)">RECORDING</span>')
    else:
        rec_html = (f'<span class="mono" style="color:var(--text-dim)">'
                    f'{snap["motion_blob_count"]} motion blob(s)</span>')

    st.markdown(f"""
    <div class="status-bar">
        <span class="status-dot {dot_class}"></span>
        <span style="font-weight:600">{snap["status_message"]}</span>
        <span class="mono" style="color:var(--text-dim); font-size:0.82rem;">{current_mode}</span>
        <span style="flex:1"></span>
        {rec_html}
    </div>
    """, unsafe_allow_html=True)

    col_video, col_alerts = st.columns([3, 2])

    with col_video:
        st.markdown('<div class="video-frame">', unsafe_allow_html=True)
        preview = snap["frame"]
        if preview is not None and preview.size > 0:
            # snapshot() already returns a fresh, exclusively-owned copy -
            # drawing straight onto it instead of copying again.
            # Amber = detection zone currently being edited (unsaved).
            cv2.rectangle(preview, (zx1, zy1), (zx2, zy2), (61, 163, 232), 2)
            # Violet = ignore zone currently being edited (unsaved), if any.
            if ix1 is not None:
                cv2.rectangle(preview, (ix1, iy1), (ix2, iy2), (224, 133, 157), 2)
            # Blue/gray (drawn by engine._on_frame) = the zones actually in effect.
            # Downscale before sending to the browser - full camera res isn't
            # needed for a live preview and just adds encode/transfer lag.
            fh, fw = preview.shape[:2]
            if fw > 480 and fh > 0:
                scale = 480 / fw
                preview = cv2.resize(preview, (480, max(1, int(fh * scale))))
            # Encode to JPEG and embed as a base64 data URI directly in the
            # markdown, INSTEAD of st.image(). st.image() (even given raw
            # bytes) registers the frame with Streamlit's in-memory media
            # file manager and points an <img> tag at a content-hash URL
            # for the browser to fetch separately - under updates this
            # rapid, that fetch was consistently losing the race against
            # the server already replacing the file with the next frame,
            # confirmed via repeated 404s in the browser console. A data
            # URI carries the image bytes inline in the same update, so
            # there's no separate request left to race.
            ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                st.markdown(
                    f'<img src="data:image/jpeg;base64,{b64}" '
                    f'style="width:100%; display:block; border-radius:4px;" />',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="mono" style="color:var(--text-dim); padding:2rem;">'
                    'Frame unavailable this refresh.</p>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<p class="mono" style="color:var(--text-dim); padding:2rem;">'
                'No feed — press Start monitoring.</p>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_alerts:
        st.markdown("**Alerts**")
        alerts = snap["alerts"]
        if not show_routine:
            alerts = [a for a in alerts if _is_notable(a)]

        if not alerts:
            st.markdown('<p class="mono" style="color:var(--text-dim)">No alerts yet.</p>',
                        unsafe_allow_html=True)
        for a in alerts[:30]:
            _render_alert(a)


live_view()

st.markdown("---")

tab_camera, tab_motion, tab_faces = st.tabs(
    ["\U0001F4F9 Camera & Schedule", "\U0001F3AF Sensitivity & Motion", "\U0001F9D1 Known Faces"]
)

# ---------------------------------------------------------------- camera & schedule --

with tab_camera:
    with st.container(border=True):
        _section_header("\U0001F4E1", "Camera source")
        st.caption("A number is a local webcam index (0 = default). Anything else is treated as an RTSP/DroidCam URL.")
        source_input = st.text_input("Camera source", value=str(engine.cfg.get("source", 0)), label_visibility="collapsed")
        if snap["running"]:
            st.caption("Monitoring is running — applying this reconnects the camera once, on purpose.")
        if st.button("Apply camera source"):
            try:
                new_source = int(source_input)
            except ValueError:
                new_source = source_input
            engine.update_source(new_source)
            _save_cfg(engine.cfg)
            st.success("Source updated.")
            st.rerun()

    with st.container(border=True):
        _section_header("\U0001F550", "AI-mode schedule")
        st.markdown(
            '<p class="section-caption">During these windows the full pipeline runs '
            '(motion \u2192 person/face \u2192 alerts). Outside them, it\'s plain '
            'motion-triggered recording — no AI, no alerts.</p>',
            unsafe_allow_html=True,
        )

        if "ai_windows" not in st.session_state:
            st.session_state.ai_windows = [
                {"id": uuid.uuid4().hex[:8], "start": w[0], "end": w[1]}
                for w in engine.cfg.get("schedule", {}).get("ai_mode", [["00:00", "23:59"]])
            ]

        current_windows = []
        for row in st.session_state.ai_windows:
            c1, c2, c3 = st.columns([3, 3, 1])
            with c1:
                s_val = st.time_input("Start", value=_parse_hhmm(row["start"]),
                                       key=f"ai_start_{row['id']}", label_visibility="collapsed")
            with c2:
                e_val = st.time_input("End", value=_parse_hhmm(row["end"]),
                                       key=f"ai_end_{row['id']}", label_visibility="collapsed")
            with c3:
                if st.button("\u2715", key=f"ai_remove_{row['id']}", help="Remove this window"):
                    st.session_state.ai_windows = [w for w in st.session_state.ai_windows if w["id"] != row["id"]]
                    st.rerun()
            current_windows.append((row["id"], s_val, e_val))

        col_add, col_save = st.columns(2)
        with col_add:
            if st.button("+ Add window", use_container_width=True):
                st.session_state.ai_windows.append({"id": uuid.uuid4().hex[:8], "start": "09:00", "end": "17:00"})
                st.rerun()
        with col_save:
            if st.button("Save AI-mode schedule", type="primary", use_container_width=True):
                windows_to_save = [[_fmt_hhmm(s_val), _fmt_hhmm(e_val)] for _, s_val, e_val in current_windows]
                if not windows_to_save:
                    st.warning("Add at least one window, or the system stays in CCTV-only mode all the time.")
                else:
                    engine.update_schedule_ai_mode(windows_to_save)
                    _save_cfg(engine.cfg)
                    st.success("AI-mode schedule saved.")

    with st.container(border=True):
        _section_header("\U0001F511", "Authorized-entry schedule")
        st.markdown(
            '<p class="section-caption">Outside these days/hours, a recognized face still isn\'t treated '
            'as authorized — nobody\'s expected, so the system skips straight to an unauthorized alert.</p>',
            unsafe_allow_html=True,
        )
        auth_cfg = engine.cfg.get("auth_schedule", {})
        auth_days = st.multiselect("Active days", options=ALL_DAYS, default=auth_cfg.get("days", ALL_DAYS))
        ac1, ac2 = st.columns(2)
        with ac1:
            auth_start = st.time_input("Start", value=_parse_hhmm(auth_cfg.get("start", "00:00")), key="auth_start")
        with ac2:
            auth_end = st.time_input("End", value=_parse_hhmm(auth_cfg.get("end", "23:59")), key="auth_end")
        if st.button("Save authorized-entry schedule"):
            engine.update_auth_schedule(auth_days, _fmt_hhmm(auth_start), _fmt_hhmm(auth_end))
            _save_cfg(engine.cfg)
            st.success("Authorized-entry schedule saved.")

# ---------------------------------------------------------------- sensitivity & motion --

with tab_motion:
    with st.container(border=True):
        _section_header("\U0001F3AF", "Detection sensitivity")
        preset_options = list(SENSITIVITY_PRESETS.keys())
        current_preset = engine.cfg.get("sensitivity", "normal")
        preset_index = preset_options.index(current_preset) if current_preset in preset_options else preset_options.index("normal")
        new_preset = st.selectbox(
            "Quick preset", options=preset_options, index=preset_index,
            format_func=lambda p: p.replace("_", " ").title(),
            help="Jumps the slider below to that preset's value - fine-tune it after if needed.",
        )
        if new_preset != current_preset:
            engine.update_sensitivity(new_preset)
            _save_cfg(engine.cfg)

        motion_threshold_val = st.slider(
            "Motion area threshold (px\u00b2)", min_value=100, max_value=6000,
            value=int(engine.cfg.get("motion_area_threshold", 1500)), step=50,
            help="Lower = trips on smaller movements. This is the actual live number - "
                 "the preset above is just a shortcut to set it.",
        )
        if motion_threshold_val != engine.cfg.get("motion_area_threshold", 1500):
            engine.update_motion_threshold(motion_threshold_val)
            _save_cfg(engine.cfg)

        ignore_pets_val = st.checkbox("Ignore small/low pets (filter cats & dogs by shape)",
                                       value=engine.cfg.get("ignore_pets", False))
        if ignore_pets_val != engine.cfg.get("ignore_pets", False):
            engine.update_ignore_pets(ignore_pets_val)
            _save_cfg(engine.cfg)

    with st.container(border=True):
        _section_header("\U0001F39A\uFE0F", "Detection thresholds")
        st.caption("How confident each model needs to be before it acts. Lower catches more but risks "
                    "more false positives; higher is stricter but can miss real events.")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            hazard_thresh = st.slider("Fire/smoke confidence", 0.05, 0.95,
                                       value=float(engine.cfg.get("hazard_confidence_threshold", 0.2)), step=0.05)
        with tc2:
            face_thresh = st.slider("Face match confidence", 0.05, 0.95,
                                     value=float(engine.cfg.get("face_match_threshold", 0.45)), step=0.05)
        with tc3:
            person_thresh = st.slider("Person-detection gate", 0.05, 0.95,
                                       value=float(engine.cfg.get("person_confidence_threshold", 0.4)), step=0.05,
                                       help="How confident a person detection needs to be before "
                                            "the face/identity pipeline runs on that clip at all.")
        if hazard_thresh != engine.cfg.get("hazard_confidence_threshold", 0.2):
            engine.update_hazard_confidence_threshold(hazard_thresh)
            _save_cfg(engine.cfg)
        if face_thresh != engine.cfg.get("face_match_threshold", 0.45):
            engine.update_face_match_threshold(face_thresh)
            _save_cfg(engine.cfg)
        if person_thresh != engine.cfg.get("person_confidence_threshold", 0.4):
            engine.update_person_threshold(person_thresh)
            _save_cfg(engine.cfg)

    with st.container(border=True):
        _section_header("\u23F1\uFE0F", "Recording timing")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            cooldown = st.number_input("Cooldown (s)", min_value=0.5, max_value=30.0,
                                        value=float(engine.cfg.get("cooldown_seconds", 2.5)), step=0.5,
                                        help="Stop recording after this much time with no motion.")
        with rc2:
            max_clip = st.number_input("Max clip length (s)", min_value=5, max_value=300,
                                        value=int(engine.cfg.get("max_clip_seconds", 45)), step=5,
                                        help="Hard cap on a single clip's length.")
        with rc3:
            pre_buffer = st.number_input("Pre-buffer (s)", min_value=0.0, max_value=10.0,
                                          value=float(engine.cfg.get("pre_buffer_seconds", 1.5)), step=0.5,
                                          help="Footage kept from just BEFORE motion is detected.")
        if (cooldown, max_clip, pre_buffer) != (
            engine.cfg.get("cooldown_seconds", 2.5), engine.cfg.get("max_clip_seconds", 45),
            engine.cfg.get("pre_buffer_seconds", 1.5),
        ):
            engine.update_recording_params(cooldown_seconds=cooldown, max_clip_seconds=max_clip, pre_buffer_seconds=pre_buffer)
            _save_cfg(engine.cfg)

    with st.container(border=True):
        _section_header("\U0001F501", "Repetitive motion filter")
        st.caption("Same spot triggering over and over (a flag flapping, a branch) gets suppressed after this many hits.")
        mc1, mc2 = st.columns(2)
        with mc1:
            rep_window = st.number_input("Window (s)", min_value=5, max_value=600,
                                          value=int(engine.cfg.get("repetitive_motion_window", 60)), step=5)
        with mc2:
            rep_max = st.number_input("Max triggers before suppressing", min_value=1, max_value=50,
                                       value=int(engine.cfg.get("repetitive_motion_max_triggers", 5)), step=1)
        if (rep_window, rep_max) != (
            engine.cfg.get("repetitive_motion_window", 60), engine.cfg.get("repetitive_motion_max_triggers", 5),
        ):
            engine.update_repetitive_motion(window=rep_window, max_triggers=rep_max)
            _save_cfg(engine.cfg)

# ---------------------------------------------------------------- known faces --

with tab_faces:
    with st.container(border=True):
        _section_header("\U0001F9D1", "Enrolled")
        names = engine.list_known_faces()
        if not names:
            st.caption("No faces enrolled yet.")
        else:
            for name in names:
                fc1, fc2 = st.columns([4, 1])
                with fc1:
                    st.write(name)
                with fc2:
                    if st.button("Remove", key=f"rm_face_{name}", use_container_width=True):
                        engine.remove_known_face(name)
                        st.rerun()

    with st.container(border=True):
        _section_header("\u2795", "Enroll a new face")
        st.markdown(
            '<p class="section-caption">Enrollment reads a photo — it never opens a second camera handle '
            'while monitoring is running. That kind of conflict is exactly what was behind the lag/'
            'close-open issue, so it\'s avoided here on purpose.</p>',
            unsafe_allow_html=True,
        )

        enroll_name = st.text_input("Name", key="enroll_name_input")
        source_choice = st.radio("Photo source", ["Upload a photo", "Use current live frame"], horizontal=True)

        photo_bgr = None
        if source_choice == "Upload a photo":
            uploaded = st.file_uploader("Photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded is not None:
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                photo_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if photo_bgr is None:
                    st.error("Could not read that image file — try a different photo.")
        else:
            if not snap["running"]:
                st.info("Start monitoring first to grab a frame from the live feed.")
            elif snap.get("raw_frame") is None:
                st.info("Waiting for the first frame...")
            else:
                photo_bgr = snap["raw_frame"]

        if photo_bgr is not None:
            st.image(photo_bgr, channels="BGR", caption="Photo to enroll", width=240)

        if st.button("Enroll", disabled=(photo_bgr is None or not enroll_name.strip())):
            with st.spinner("Loading the face recognition model if needed, then enrolling..."):
                success = engine.enroll_face(enroll_name.strip(), photo_bgr)
            if success:
                st.success(f"Enrolled '{enroll_name.strip()}'.")
                st.rerun()
            else:
                st.error("No face found in that photo — try a clearer, front-facing shot.")