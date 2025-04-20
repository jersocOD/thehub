# main.py
# FPV Multi-Drone Control with Enhanced UI/UX for multi-drone missions

import os
import socket
import threading
import time

import av
import cv2
import numpy as np
import torch
from flask import Flask, Response, request, jsonify

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
# For demonstration, this script is set up to control a single Tello,
# designated as drone1. Additional drones (drone2, drone3, etc.) are
# shown in the UI but use placeholder logic. Adapt as needed.

# Known Tello parameters
DRONE1_IP           = '192.168.10.1'  # Tello IP
COMMAND_PORT        = 8889
VIDEO_PORT          = 11111
LOCAL_COMMAND_PORT  = 9000

# Frame rate & detection parameters
TARGET_FPS          = 30
DETECT_FPS          = 5     # run person detection 5×/sec
CENTER_TOL          = 0.10  # ±10% of frame width for horizontal offset
SIZE_THRESH         = 0.20  # box ≥20% of frame width → "close enough"

# -------------------------------------------------------------------
# GLOBAL STATE
# -------------------------------------------------------------------

# Example multi-drone “fleet” definition
# You can expand IP, ports, etc. for each drone in your real environment
DRONES = {
    'drone1': {
        'name': 'Drone 1',
        'ip': DRONE1_IP,
        'command_port': COMMAND_PORT,
        'local_port': LOCAL_COMMAND_PORT,
        'video_port': VIDEO_PORT,
        'mode': 'FPV',       # or 'Autonomous'
        'track_person': False,
        'last_detect_time': 0.0,
        'latest_frame': None
    },
    'drone2': {
        'name': 'Drone 2',
        'ip': '192.168.10.2',    # Placeholder
        'command_port': 8889,    # Placeholder
        'local_port': 9001,      # Placeholder
        'video_port': 11112,     # Placeholder
        'mode': 'Autonomous',
        'track_person': False,
        'last_detect_time': 0.0,
        'latest_frame': None
    },
    'drone3': {
        'name': 'Drone 3',
        'ip': '192.168.10.3',    # Placeholder
        'command_port': 8890,    # Placeholder
        'local_port': 9002,      # Placeholder
        'video_port': 11113,     # Placeholder
        'mode': 'Autonomous',
        'track_person': False,
        'last_detect_time': 0.0,
        'latest_frame': None
    },
}

# Active drone for FPV streaming/commands
ACTIVE_DRONE_ID = 'drone1'

# -------------------------------------------------------------------
# LOAD YOLOv5 MODEL
# -------------------------------------------------------------------
base_dir = os.path.dirname(__file__)
repo_dir = os.path.join(base_dir, 'yolov5')
weights  = os.path.join(base_dir, 'yolov5s.pt')
if not os.path.isdir(repo_dir):
    raise FileNotFoundError(f"Missing yolov5 repo at {repo_dir}")
if not os.path.isfile(weights):
    raise FileNotFoundError(f"Missing weights at {weights}")

os.environ['YOLOv5_SKIP_UPDATE'] = '1'  # skip auto-update in torch.hub
model = torch.hub.load(repo_dir, 'custom', path=weights, source='local')
model.conf = 0.5  # confidence threshold

# -------------------------------------------------------------------
# SETUP UDP COMMAND SOCKET FOR THE PRIMARY DRONE (drone1)
# -------------------------------------------------------------------
command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
command_sock.bind(('', DRONES['drone1']['local_port']))
command_sock.settimeout(1)

def send_command(cmd: str):
    """Send a single SDK command to the currently active Tello (drone1)."""
    global ACTIVE_DRONE_ID
    drone = DRONES[ACTIVE_DRONE_ID]
    if ACTIVE_DRONE_ID != 'drone1':
        # For demonstration, we show placeholders for other drones
        print(f"[MOCK COMMAND to {drone['name']}] {cmd}")
        return

    print(f"[SEND to {drone['name']}] {cmd}")
    try:
        command_sock.sendto(cmd.encode('utf-8'), (drone['ip'], drone['command_port']))
        resp, _ = command_sock.recvfrom(1024)
        print(f"[RECV] {resp.decode()}")
    except socket.timeout:
        print(f"[TIMEOUT] No response for {cmd}")
    except Exception as e:
        print(f"[ERROR] {cmd} → {e}")

# Initialize Tello connection (only for drone1 in this example)
for c in ("command", "streamon"):
    send_command(c)

# -------------------------------------------------------------------
# FLASK APP
# -------------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------------
# VIDEO RECEIVING LOGIC (drone1 only)
# -------------------------------------------------------------------
video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_socket.bind(('', DRONES['drone1']['video_port']))

frame_lock = threading.Lock()

def receive_video():
    """Background thread: read H.264 UDP from Tello (drone1) & store latest_frame."""
    container = av.open(video_socket.makefile('rb'))
    for packet in container.demux():
        for frame in packet.decode():
            img = frame.to_ndarray(format='bgr24')
            with frame_lock:
                DRONES['drone1']['latest_frame'] = img

# Start the background thread for drone1 video
threading.Thread(target=receive_video, daemon=True).start()


# -------------------------------------------------------------------
# STREAM GENERATOR (MJPEG)
# -------------------------------------------------------------------
def generate_mjpeg():
    """Yield MJPEG frames for the *active* drone. If active=drone1, show real feed;
       otherwise show a placeholder image or blank frame for demonstration."""
    global ACTIVE_DRONE_ID
    interval = 1.0 / TARGET_FPS

    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "No real feed for this drone", (30,240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    while True:
        t0 = time.time()

        # Decide which frame source to use
        if ACTIVE_DRONE_ID == 'drone1':
            # Pull from the real Tello feed
            with frame_lock:
                frame = DRONES['drone1']['latest_frame'].copy() \
                         if DRONES['drone1']['latest_frame'] is not None else None
        else:
            # Mock placeholder
            frame = placeholder.copy()

        if frame is not None:
            # If we are in track_person mode for the active drone, run detection
            drone = DRONES[ACTIVE_DRONE_ID]
            annotated = frame.copy()
            now = time.time()

            if drone['track_person'] and (now - drone['last_detect_time']) >= (1.0 / DETECT_FPS):
                # Update detection timestamp
                drone['last_detect_time'] = now

                H, W = frame.shape[:2]
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

                if people and ACTIVE_DRONE_ID == 'drone1':
                    # For demonstration, only drone1 can move/rotate
                    x1, y1, x2, y2, _ = max(people, key=lambda x: x[4])
                    cx = (x1 + x2) / 2
                    bw = (x2 - x1)
                    offset = cx / W - 0.5
                    if abs(offset) > CENTER_TOL:
                        cmd = 'cw 15' if offset > 0 else 'ccw 15'
                        send_command(cmd)
                    elif (bw/W) < SIZE_THRESH:
                        send_command('forward 30')
                    else:
                        # Reached subject
                        send_command('forward 30')  # example
                        print("Reached subject.")

                    # Draw bounding box
                    cv2.rectangle(annotated, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(annotated, "person",
                                (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                annotated = frame

            display = cv2.resize(annotated, (640, 480))
            ret, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')
        else:
            # No frame, yield black
            black_frame = np.zeros((480,640,3), np.uint8)
            ret, buf = cv2.imencode('.jpg', black_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')

        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.route('/')
def index():
    """
    Main UI: 
    - Sidebar with drone list, quick controls 
    - Header with mode toggle 
    - Video + Telemetry HUD 
    - Mission/Map panel 
    """
    # Build dynamic drone list for the sidebar
    drone_list_html = []
    for d_id, d_info in DRONES.items():
        active_class = 'active' if d_id == ACTIVE_DRONE_ID else ''
        drone_list_html.append(f'''
          <li class="drone-item {active_class}" onclick="selectDrone('{d_id}')">
            {d_info['name']} <span style="font-size:0.8em;">({d_info['mode']})</span>
          </li>''')
    drone_list_str = "\n".join(drone_list_html)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FPV Multi-Drone Control</title>
  <style>
    :root {{
      --accent: #4f46e5;
      --bg: #0f172a;
      --btn-bg: #1e293b;
      --btn-border: #475569;
      --btn-hover: #334155;
      --text: #f8fafc;
      --highlight: #10b981;
      --danger: #ef4444;
      --sidebar-w: 280px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      display: grid;
      grid-template-columns: var(--sidebar-w) 1fr;
      height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', sans-serif;
      overflow: hidden;
    }}
    /* Sidebar */
    .sidebar {{
      background: var(--btn-bg);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .sidebar h2 {{ font-size: 1.2rem; margin-bottom: 6px; }}
    .drone-list {{ list-style: none; display: flex; flex-direction: column; gap: 6px; }}
    .drone-item {{
      padding: 8px;
      border: 2px solid var(--btn-border);
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.2s;
    }}
    .drone-item.active,
    .drone-item:hover {{
      background: var(--highlight);
      color: var(--bg);
    }}
    .control-grid {{ 
      display: grid; 
      grid-template-columns: 1fr 1fr; 
      gap: 8px; 
      margin-top: 8px; 
    }}
    .control-grid button {{
      padding: 10px;
      font-size: 0.9rem;
      background: var(--btn-bg);
      color: var(--text);
      border: 2px solid var(--btn-border);
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.2s;
    }}
    .control-grid button:hover {{ background: var(--btn-hover); }}
    .takeoff {{ background: var(--highlight); border-color: var(--highlight); }}
    .land    {{ background: var(--danger);    border-color: var(--danger);    }}
    /* Main Content */
    .main-content {{ display: flex; flex-direction: column; }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--btn-border);
    }}
    .header h1 {{ font-size: 1.4rem; }}
    .header button {{
      padding: 6px 12px;
      background: var(--btn-bg);
      border: 2px solid var(--btn-border);
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.2s;
      color: var(--text);
    }}
    .header button:hover {{ background: var(--btn-hover); }}
    .video-hud {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 12px;
      flex: 1;
      padding: 16px;
    }}
    .video-container {{
      position: relative;
      background: black;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
    }}
    .video-container img {{ width: 100%; height: auto; object-fit: contain; }}
    .telemetry-overlay {{
      position: absolute;
      top: 12px;
      left: 12px;
      background: rgba(0,0,0,0.5);
      padding: 8px;
      border-radius: 6px;
      display: flex;
      gap: 12px;
      font-size: 0.85rem;
    }}
    .map-container {{
      background: #1e293b;
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      justify-content: start;
      align-items: center;
      color: var(--btn-border);
      font-size: 1rem;
      padding: 12px;
      gap: 12px;
    }}
    .map-container h2 {{ margin-bottom: 8px; color: var(--text); }}
    .mission-item {{
      background: var(--btn-bg);
      border: 2px solid var(--btn-border);
      border-radius: 6px;
      padding: 6px;
      width: 100%;
      text-align: center;
      margin-bottom: 6px;
      cursor: pointer;
      transition: background 0.2s;
      color: var(--text);
    }}
    .mission-item:hover {{
      background: var(--btn-hover);
    }}
  </style>
</head>
<body>
  <aside class="sidebar">
    <h2>Drone Fleet</h2>
    <ul class="drone-list">
      {drone_list_str}
    </ul>
    <div class="control-grid">
      <button class="takeoff" onclick="sendSDKCommand('takeoff')">🚀 Takeoff</button>
      <button class="land" onclick="sendSDKCommand('land')">🛬 Land</button>
      <button onclick="sendSDKCommand('forward 30')">↑ Fwd</button>
      <button onclick="sendSDKCommand('back 30')">↓ Back</button>
      <button onclick="sendSDKCommand('left 30')">← Left</button>
      <button onclick="sendSDKCommand('right 30')">→ Right</button>
      <button onclick="sendSDKCommand('up 30')">🔼 Up</button>
      <button onclick="sendSDKCommand('down 30')">🔽 Down</button>
      <button onclick="sendSDKCommand('ccw 15')">⟲ Rot L</button>
      <button onclick="sendSDKCommand('cw 15')">⟳ Rot R</button>
      <button onclick="sendSDKCommand('flip f')">↩️ Flip</button>
      <button onclick="sendSDKCommand('emergency')">🛑 Emerg</button>
      <button onclick="sendSDKCommand('battery?')">🔋 Batt?</button>
      <button onclick="sendSDKCommand('command')">🔄 Reconnect</button>
    </div>
  </aside>
  <main class="main-content">
    <header class="header">
      <h1>FPV Multi-Drone Control</h1>
      <button id="modeToggleBtn" onclick="toggleMode()"></button>
    </header>
    <section class="video-hud">
      <div class="video-container">
        <img id="droneStream" src="/video_feed" alt="FPV Stream">
        <div class="telemetry-overlay">
          <div>Alt: <span id="altitude">--</span> cm</div>
          <div>Bat: <span id="battery">--</span>%</div>
          <div>Spd: <span id="speed">--</span> cm/s</div>
        </div>
      </div>
      <div class="map-container">
        <h2>Mission Control</h2>
        <div style="font-size:0.9em;color:var(--text);margin-bottom:4px;">
          Manage multi-drone flight paths & missions
        </div>
        <!-- Example placeholder mission items -->
        <div class="mission-item" onclick="alert('Plan Mission A')">Mission A</div>
        <div class="mission-item" onclick="alert('Plan Mission B')">Mission B</div>
        <div class="mission-item" onclick="alert('Plan Mission C')">Mission C</div>
      </div>
    </section>
  </main>

  <script>
    let activeDrone = "{ACTIVE_DRONE_ID}";
    let drones = {{"drone1": "FPV", "drone2": "Autonomous", "drone3": "Autonomous"}};

    // Initialize the Mode Toggle Button text
    function updateModeToggleText() {{
      const btn = document.getElementById('modeToggleBtn');
      const currentMode = drones[activeDrone];
      if (currentMode === 'FPV') {{
        btn.textContent = 'Switch to Autonomous Mode';
      }} else {{
        btn.textContent = 'Switch to FPV Mode';
      }}
    }}
    updateModeToggleText();

    // Switch the active drone
    function selectDrone(droneId) {{
      fetch('/select_drone/' + droneId, {{method:'POST'}})
        .then(r=>r.json()).then(data=>{{
          activeDrone = data.activeDrone;
          // Update UI
          const items = document.querySelectorAll('.drone-item');
          items.forEach(item=>item.classList.remove('active'));
          const activeItem = document.querySelector(`[onclick="selectDrone('{droneId}')"]`);
          if(activeItem) activeItem.classList.add('active');
          // Refresh mode toggle text
          updateModeToggleText();
        }});
    }}

    // Toggle mode for the active drone (FPV <-> Autonomous)
    function toggleMode() {{
      fetch('/toggle_mode/'+activeDrone, {{method:'POST'}})
      .then(r=>r.json())
      .then(data=> {{
        drones[activeDrone] = data.newMode;
        updateModeToggleText();
      }});
    }}

    // Send an SDK command (like 'takeoff') to the active drone
    function sendSDKCommand(cmd) {{
      const formData = new FormData();
      formData.append('cmd', cmd);
      fetch('/command', {{method:'POST', body: formData}});
    }}

    // Example: you could periodically fetch telemetry here & update #altitude, #battery, #speed
    // For demonstration, we'll omit real Tello states unless you expand with Tello's SDK queries.
  </script>
</body>
</html>'''


@app.route('/video_feed')
def video_feed():
    """Video feed route: streams MJPEG from generate_mjpeg()."""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/command', methods=['POST'])
def handle_command():
    """Send a direct SDK command (POST form-data cmd=...) to the active drone."""
    cmd = request.form.get('cmd')
    if cmd:
        send_command(cmd)
    return ('', 204)


@app.route('/track_person', methods=['POST'])
def track_person_route():
    """Enable person-tracking on the active drone."""
    global ACTIVE_DRONE_ID
    DRONES[ACTIVE_DRONE_ID]['track_person'] = True
    return ('', 204)


@app.route('/takeoff', methods=['POST'])
def takeoff():
    """Shortcut to 'takeoff'."""
    send_command("takeoff")
    return ('', 204)


@app.route('/land', methods=['POST'])
def land():
    """Shortcut to 'land'."""
    send_command("land")
    return ('', 204)


@app.route('/select_drone/<drone_id>', methods=['POST'])
def select_drone(drone_id):
    """Switch active drone for streaming and commands."""
    global ACTIVE_DRONE_ID
    if drone_id in DRONES:
        ACTIVE_DRONE_ID = drone_id
    return jsonify({"activeDrone": ACTIVE_DRONE_ID})


@app.route('/toggle_mode/<drone_id>', methods=['POST'])
def toggle_mode(drone_id):
    """Toggle a drone's mode between FPV and Autonomous."""
    if drone_id not in DRONES:
        return jsonify({"error": "Invalid drone"}), 400

    current_mode = DRONES[drone_id]['mode']
    new_mode = 'FPV' if current_mode == 'Autonomous' else 'Autonomous'
    DRONES[drone_id]['mode'] = new_mode

    # If turning FPV ON for a different drone, turn OFF FPV for others
    if new_mode == 'FPV':
        for d_id in DRONES:
            if d_id != drone_id:
                DRONES[d_id]['mode'] = 'Autonomous'

    return jsonify({"newMode": new_mode})


# -------------------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Start Flask
    # Access via http://<server>:5005
    app.run(host='0.0.0.0', port=5005, debug=False)
