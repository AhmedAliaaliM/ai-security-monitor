# Dashboard

`app.py` replaces `python main.py --preview` with a browser UI: live feed,
a hazard/person alert feed, a visual detection/ignore-zone editor, and a
settings area covering everything `main.py`/`my_camera.json` support —
camera source, both schedules, sensitivity and motion tuning, and known-face
enrollment. It reuses `core/` unchanged — `engine.py` is the only new glue
code, adapting the same callbacks `main.py` uses into a background thread +
shared state that `app.py` polls.

**If you hit laggy video or the camera repeatedly closing/reopening through
the app (but not through `main.py`):** that was a real bug, now fixed. It
came from `hazard_check_interval` being set very low (e.g. via the sidebar
slider) — each hazard check runs the fire-detection model (and sometimes the
low-light enhancer) in a background thread, and when checks were spawned
faster than they could finish, they piled up and starved the capture loop of
CPU, causing camera reads to fail and reconnect on a loop. `core/capture.py`
now refuses to start a new hazard check while one is still running, so it
self-throttles to whatever the model can actually keep up with instead of
piling up threads — any interval is safe now, including the sidebar
slider's minimum.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. It reads/writes `my_camera.json` in this
folder, same as `main.py --config my_camera.json`.

## The zone editors (sidebar)

**Detection zone** — drag the Left/Right/Top/Bottom sliders; an amber box
previews the change live on the feed. Click **Save zone** to apply it
immediately (no restart) and write it to `my_camera.json`. The blue box
on the feed is the zone actually in effect; amber is your unsaved edit.
Only motion inside this box is watched, recorded, or scored at all.

**Ignore zone** (in the "Ignore zone (optional)" expander) — same idea,
but for a spot to *exclude* — a ceiling fan, a tree branch, anything that
keeps false-triggering. Previews in violet before you save; the zone
actually in effect is drawn in gray. Motion here is dropped before it's
even considered, regardless of the detection zone.

Both editors push straight into the running capture loop's live motion
gating the moment you click Save — not just the visual preview and not
just clip scoring afterward, so what you see previewed is what actually
gates recording from that point on.

## Settings tabs (below the live feed)

- **Camera & Schedule** — camera source (webcam index or an RTSP/DroidCam
  URL; applying it while running triggers one deliberate reconnect, not the
  repeated involuntary kind the fix above prevents); the AI-mode schedule
  (add/remove time windows — full pipeline runs inside them, plain
  motion-triggered recording with no AI/alerts outside them); the
  authorized-entry schedule (days + hours a recognized face counts as
  authorized rather than triggering an unauthorized alert).
- **Sensitivity & Motion** — the `high_security` / `normal` / `low` preset,
  the pets filter, recording timing (cooldown, max clip length, pre-buffer),
  and the repetitive-motion suppression window/threshold.
- **Known Faces** — lists everyone in `known_faces.json` with a Remove
  button each, plus enrollment: give it a name and either upload a photo or
  grab the current live frame. Enrollment deliberately never opens a second
  `cv2.VideoCapture` — it reuses the already-open feed (or a file) — since a
  second camera handle competing with the running one is exactly the kind of
  conflict behind the lag/close-open bug above.

Every setting here writes straight into `my_camera.json` and applies
immediately if monitoring is already running — none of them need a restart.

## Deploying — read this before hosting it anywhere

This app opens your webcam directly (`cv2.VideoCapture`), the same way
`main.py` does. That only works on a machine that **physically has the
camera attached**. Cloud hosts (Streamlit Community Cloud, Heroku, a
generic VPS, etc.) don't have your webcam, so `cv2.VideoCapture(0)` will
fail there — deploying this app's current form to the cloud will not work
for a local USB webcam, and processing a live face-recognition feed on
someone else's server isn't something to do casually with home footage
anyway.

**What actually works:**

1. **Run it on the machine with the camera**, exactly like you do now.
   `streamlit run app.py` starts a local web server — that's the "deploy"
   step. Everything (camera access, model inference, your face database)
   stays on that machine.

2. **View it from other devices on your home WiFi:**
   ```
   streamlit run app.py --server.address 0.0.0.0
   ```
   Then visit `http://<that-machine's-LAN-IP>:8501` from your phone or
   another computer on the same network. Find the LAN IP with `ipconfig`
   (Windows) — look for the IPv4 address.

3. **View it from outside your home network:** don't expose port 8501
   directly to the internet — that puts a live camera feed + face
   recognition behind no authentication. Use a private tunnel instead,
   e.g. [Tailscale](https://tailscale.com/) (installs on both the camera
   machine and your phone, gives you a private address that works from
   anywhere without opening any ports or exposing anything publicly).

4. **Keep it running persistently:** on Windows, a Scheduled Task that
   runs `streamlit run app.py` at login/boot is the simplest option; for
   something closer to a real service, look at
   [NSSM](https://nssm.cc/) to run it as a Windows service. Happy to set
   either of these up if you want.

If you specifically want the *processing* to run somewhere other than
this machine (e.g. a Raspberry Pi always-on box instead of your main PC),
that's a different, simpler swap — same code, just run it on that box
instead — and doesn't require any cloud service at all.
