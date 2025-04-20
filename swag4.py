# main.py
# FPV Multi-Drone Control with Enhanced UI
import os
import socket
import threading
import time

import av
import cv2
import numpy as np
import torch
from flask import Flask, Response, request

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
Tello_IP           = '192.168.10.1'
COMMAND_PORT       = 8889
VIDEO_PORT         = 11111
LOCAL_COMMAND_PORT = 9000
TARGET_FPS         = 30
DETECT_FPS         = 5     # run person detection 5×/sec
CENTER_TOL         = 0.10  # ±10% of frame width
SIZE_THRESH        = 0.20  # box ≥20% of frame width → close enough

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
track_person     = False
last_detect_time = 0.0

# -------------------------------------------------------------------
# Load YOLOv5 model locally
# -------------------------------------------------------------------
base_dir = os.path.dirname(__file__)
repo_dir = os.path.join(base_dir, 'yolov5')
weights  = os.path.join(base_dir, 'yolov5s.pt')
if not os.path.isdir(repo_dir):
    raise FileNotFoundError(f"Missing yolov5 repo at {repo_dir}")
if not os.path.isfile(weights):
    raise FileNotFoundError(f"Missing weights at {weights}")
os.environ['YOLOv5_SKIP_UPDATE'] = '1'
model = torch.hub.load(repo_dir, 'custom', path=weights, source='local')
model.conf = 0.5  # confidence threshold

# -------------------------------------------------------------------
# UDP COMMAND SOCKET
# -------------------------------------------------------------------
command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
command_sock.bind(('', LOCAL_COMMAND_PORT))
command_sock.settimeout(1)

def send_command(cmd: str):
    """Send a single SDK command to the Tello."""
    print(f"[SEND] {cmd}")
    try:
        command_sock.sendto(cmd.encode('utf-8'), (Tello_IP, COMMAND_PORT))
        resp, _ = command_sock.recvfrom(1024)
        print(f"[RECV] {resp.decode()}")
    except socket.timeout:
        print(f"[TIMEOUT] No response for {cmd}")
    except Exception as e:
        print(f"[ERROR] {cmd} → {e}")

# Initialize connection and streaming
for c in ("command", "streamon"):
    send_command(c)

# -------------------------------------------------------------------
# Flask app & video receiver
# -------------------------------------------------------------------
app = Flask(__name__)
video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_socket.bind(('', VIDEO_PORT))

frame_lock   = threading.Lock()
latest_frame = None

def receive_video():
    """Background thread: read H.264 UDP & update latest_frame."""
    global latest_frame
    container = av.open(video_socket.makefile('rb'))
    for packet in container.demux():
        for frame in packet.decode():
            img = frame.to_ndarray(format='bgr24')
            with frame_lock:
                latest_frame = img

def generate_mjpeg():
    """Yield MJPEG frames; run detection+steering when track_person=True."""
    global last_detect_time, track_person
    interval = 1.0 / TARGET_FPS
    while True:
        t0 = time.time()
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            H, W = frame.shape[:2]
            annotated = frame.copy()
            now = time.time()
            if track_person and (now - last_detect_time) >= 1.0/DETECT_FPS:
                last_detect_time = now
                dw = 640
                dh = int(H * dw / W)
                small = cv2.resize(frame, (dw, dh))
                results = model(small)
                dets = results.xyxy[0].cpu().numpy()
                people = []
                fx, fy = W/dw, H/dh
                for x1, y1, x2, y2, conf, cls in dets:
                    if int(cls) == 0 and conf > model.conf:
                        people.append((x1*fx, y1*fy, x2*fx, y2*fy, conf))
                if people:
                    x1, y1, x2, y2, _ = max(people, key=lambda x: x[4])
                    cx = (x1 + x2) / 2
                    bw = (x2 - x1)
                    offset = cx/W - 0.5
                    if abs(offset) > CENTER_TOL:
                        cmd = 'cw 15' if offset > 0 else 'ccw 15'
                        send_command(cmd)
                    elif (bw/W) < SIZE_THRESH:
                        send_command('forward 30')
                    else:
                        send_command('forward 30')
                        print("Reached subject.")
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(annotated, "person", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            display = cv2.resize(annotated, (640, 480))
            ret, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🚁 SwarmTag FPV Dashboard</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">

  <style>
    :root {
      --clr-bg: #0f172a;
      --clr-panel: #1e293b;
      --clr-accent: #4f46e5;
      --clr-text: #f8fafc;
      --clr-btn: #334155;
      --clr-btn-hover: #475569;
      --clr-green: #10b981;
      --clr-red: #ef4444;
      --hud-bg: rgba(15,23,42,0.6);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      display: grid;
      grid-template-rows: auto 1fr auto;
      grid-template-columns: 1fr;
      background: var(--clr-bg);
      color: var(--clr-text);
      font-family: Inter, sans-serif;
      height: 100vh;
      overflow: hidden;
    }
    header, footer {
      background: var(--clr-panel);
      padding: 0.5rem 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    header h1 { font-size: 1.5rem; }
    header .mode-switch {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    header .mode-switch input { width: 1.5rem; height: 1.5rem; }
    main {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 1rem;
      padding: 1rem;
      overflow: hidden;
    }
    /* Video & HUD */
    .video-panel {
      position: relative;
      background: black;
      border-radius: 8px;
      overflow: hidden;
    }
    .video-panel img {
      width: 100%; height: 100%; object-fit: contain;
    }
    .hud {
      position: absolute;
      top: 0; left: 0; right: 0;
      display: flex; justify-content: space-between;
      background: var(--hud-bg);
      padding: 0.5rem 1rem;
      font-size: 0.9rem;
    }
    .hud span { font-weight: 500; }

    /* Control Panel */
    .control-panel {
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 1rem;
    }
    .group {
      background: var(--clr-panel);
      padding: 0.75rem;
      border-radius: 6px;
    }
    .group h2 {
      font-size: 1.1rem;
      margin-bottom: 0.5rem;
      border-bottom: 1px solid var(--clr-btn);
      padding-bottom: 0.25rem;
    }
    .btn-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.5rem;
    }
    .btn-grid button {
      background: var(--clr-btn);
      color: var(--clr-text);
      border: 2px solid var(--clr-btn-hover);
      border-radius: 6px;
      padding: 0.75rem;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
    }
    .btn-grid button:hover {
      background: var(--clr-btn-hover);
      transform: translateY(-1px);
    }
    .btn-wide { grid-column: span 2; }
    .takeoff { background: var(--clr-green); border-color: var(--clr-green); }
    .land    { background: var(--clr-red);   border-color: var(--clr-red); }
    .emergency {
      background: #b91c1c; border-color: #b91c1c;
      animation: pulse 1s infinite;
    }
    @keyframes pulse {
      0%,100% { transform: scale(1); }
      50%     { transform: scale(1.05); }
    }
  </style>

  <script>
    function sendForm(cmd, formId) {
      document.getElementById(formId + '-cmd').value = cmd;
      document.getElementById(formId).submit();
    }
    function toggleTrack(el) {
      fetch('/track_person', {method:'POST'});
      el.textContent = el.checked ? '🔒 Tracking ON' : '👤 Track Person';
    }
  </script>
</head>

<body>
  <header>
    <h1>🚁 SwarmTag FPV Dashboard</h1>
    <div class="mode-switch">
      <label>
        <input type="checkbox" onchange="toggleTrack(this)" />
        👤 Track Person
      </label>
      <span id="telemetry">Battery: --% | Drones: 1</span>
    </div>
  </header>

  <main>
    <!-- Video & HUD -->
    <div class="video-panel">
      <img src="/video_feed" alt="Drone FPV feed" />
      <div class="hud">
        <span>Mode: <strong id="mode">Manual</strong></span>
        <span>FPS: <strong id="fps">--</strong></span>
        <span>Status: <strong id="status">Idle</strong></span>
      </div>
    </div>

    <!-- Control Panel -->
    <div class="control-panel">
      <div class="group">
        <h2>Movement</h2>
        <div class="btn-grid">
          <form id="fmv" action="/command" method="post"><input id="fmv-cmd" type="hidden" name="cmd"/></form>
          <button onclick="sendForm('forward 50','fmv')">↑ Forward</button>
          <button onclick="sendForm('back 50','fmv')">↓ Back</button>
          <button onclick="sendForm('left 50','fmv')">← Left</button>
          <button onclick="sendForm('right 50','fmv')">→ Right</button>
        </div>
      </div>

      <div class="group">
        <h2>Altitude & Rotate</h2>
        <div class="btn-grid">
          <form id="falt" action="/command" method="post"><input id="falt-cmd" type="hidden" name="cmd"/></form>
          <button onclick="sendForm('up 50','falt')">🔼 Up</button>
          <button onclick="sendForm('down 50','falt')">🔽 Down</button>
          <button onclick="sendForm('ccw 45','falt')">⟲ Rotate Left</button>
          <button onclick="sendForm('cw 45','falt')">⟳ Rotate Right</button>
        </div>
      </div>

      <div class="group">
        <h2>Takeoff / Land</h2>
        <div class="btn-grid">
          <form id="ftl" action="/takeoff" method="post"></form>
          <form id="fld" action="/land"    method="post"></form>
          <button class="takeoff btn-wide" onclick="document.getElementById('ftl').submit()">🚀 Takeoff</button>
          <button class="land    btn-wide" onclick="document.getElementById('fld').submit()">🛬 Land</button>
        </div>
      </div>

      <div class="group">
        <h2>Flips</h2>
        <div class="btn-grid">
          <form id="ffl" action="/command" method="post"><input id="ffl-cmd" type="hidden" name="cmd"/></form>
          <button onclick="sendForm('flip l','ffl')">↩️ Flip Left</button>
          <button onclick="sendForm('flip r','ffl')">↪️ Flip Right</button>
          <button onclick="sendForm('flip f','ffl')">⤴️ Flip Forward</button>
          <button onclick="sendForm('flip b','ffl')">⤵️ Flip Back</button>
        </div>
      </div>

      <div class="group">
        <h2>Utilities</h2>
        <div class="btn-grid">
          <form id="fut" action="/command" method="post"><input id="fut-cmd" type="hidden" name="cmd"/></form>
          <button onclick="sendForm('speed','fut')">🏎️ Speed</button>
          <button onclick="sendForm('battery?','fut')">🔋 Battery</button>
          <button onclick="sendForm('emergency','fut')" class="emergency">🛑 Emergency</button>
          <button onclick="sendForm('command','fut')">🔄 Reconnect</button>
        </div>
      </div>
    </div>
  </main>

  <footer>
    <span>© 2025 SwarmTag Hackathon Team</span>
    <span>Powered by Flask • YOLOv5 • Tello SDK</span>
  </footer>
</body>
</html>
'''


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/track_person', methods=['POST'])
def track_person_route():
    global track_person
    track_person = True
    return ('', 204)

@app.route('/takeoff', methods=['POST'])
def takeoff():
    send_command("takeoff")
    return ('', 204)

@app.route('/land', methods=['POST'])
def land():
    send_command("land")
    return ('', 204)

@app.route('/command', methods=['POST'])
def handle_command():
    cmd = request.form.get('cmd')
    if cmd:
        send_command(cmd)
    return ('', 204)

if __name__ == '__main__':
    threading.Thread(target=receive_video, daemon=True).start()
    app.run(host='0.0.0.0', port=5005, debug=False)  # HTML/UI updated inline, commands per SDK citeturn0file0
