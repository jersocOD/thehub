# main.py
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
model = torch.hub.load(repo_dir, 'custom', path=weights, source='local')  # :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
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

# -------------------------------------------------------------------
# Start drone video stream
# -------------------------------------------------------------------
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

                # downscale for speed
                dw = 640
                dh = int(H * dw / W)
                small = cv2.resize(frame, (dw, dh))

                # run YOLOv5
                results = model(small)
                dets = results.xyxy[0].cpu().numpy()

                # filter for people (class 0)
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

                    # steering logic
                    if abs(offset) > CENTER_TOL:
                        cmd = 'cw 15' if offset > 0 else 'ccw 15'
                        send_command(cmd)
                    elif (bw/W) < SIZE_THRESH:
                        send_command('forward 30')
                    else:
                        send_command('forward 30')
                        print("Reached subject.")

                    # draw box
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(annotated, "person", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # encode & yield
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
    <html>
    <head>
        <title>Tello Drone UI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            :root {
                --accent: #4f46e5; --bg: #0f172a; --btn-bg: #1e293b;
                --btn-hover: #334155; --btn-border: #475569; --text: #f8fafc;
                --red: #ef4444; --blue: #3b82f6; --green: #10b981;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Inter', sans-serif;
                background: var(--bg); color: var(--text);
                height: 100vh; display: flex; flex-direction: column;
                padding: 10px; overflow: hidden;
            }
            h1 { text-align: center; margin: 5px 0; font-size: 1.8rem; }
            .main-grid {
                display: grid; grid-template-rows: 1fr auto;
                height: 100%; gap: 10px;
            }
            .video-container {
                background: black; border-radius: 8px;
                overflow: hidden; display: flex;
                justify-content: center; align-items: center;
            }
            .video-container img {
                max-width: 100%; max-height: 100%;
                object-fit: contain;
            }
            .control-grid {
                display: grid; grid-template-columns: repeat(4, 1fr);
                gap: 8px;
            }
            .control-grid button {
                width: 100%; height: 100%; min-height: 60px;
                font-size: 1.2rem; padding: 10px;
                background: var(--btn-bg); color: var(--text);
                border: 2px solid var(--btn-border);
                border-radius: 8px; font-weight: bold;
                cursor: pointer; transition: all 0.2s;
            }
            .control-grid button:hover {
                background: var(--btn-hover); transform: translateY(-2px);
            }
            .double-width { grid-column: span 2; }
            .takeoff { background: var(--green); border-color: var(--green); }
            .land    { background: var(--red);   border-color: var(--red);   }
            .emergency {
                background: #b91c1c; border-color: #b91c1c;
                animation: pulse 1s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
        </style>
    </head>
    <body>
        <div class="main-grid">
            <div class="video-container">
                <img src="/video_feed">
            </div>
            <div class="control-grid">
                <!-- Movement -->
                <form action="/command" method="post"><input type="hidden" name="cmd" value="forward 50"><button>↑ Forward</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="left 50"><button>← Left</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="right 50"><button>→ Right</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="back 50"><button>↓ Back</button></form>
                <!-- Altitude -->
                <form action="/command" method="post"><input type="hidden" name="cmd" value="up 50"><button>🔼 Up</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="down 50"><button>🔽 Down</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="ccw 45"><button>⟲ Rotate Left</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="cw 45"><button>⟳ Rotate Right</button></form>
                <!-- Takeoff/Land -->
                <form action="/takeoff" method="post"><button class="takeoff double-width">🚀 Takeoff</button></form>
                <form action="/land"    method="post"><button class="land    double-width">🛬 Land</button></form>
                <!-- Flips -->
                <form action="/command" method="post"><input type="hidden" name="cmd" value="flip l"><button>↩️ Flip Left</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="flip r"><button>↪️ Flip Right</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="flip f"><button>⤴️ Flip Forward</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="flip b"><button>⤵️ Flip Back</button></form>
                <!-- Utility -->
                <form action="/command" method="post"><input type="hidden" name="cmd" value="speed"><button>🏎️ Speed</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="battery?"><button>🔋 Battery</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="emergency"><button class="emergency">🛑 Emergency Stop</button></form>
                <form action="/command" method="post"><input type="hidden" name="cmd" value="command"><button>Reconnect</button></form>
                <!-- NEW: Track Person -->
                <form action="/track_person" method="post">
                  <button style="background:#10b981;color:#0f172a;font-size:1.1rem;">
                    👤 Track Person
                  </button>
                </form>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    app.run(host='0.0.0.0', port=5005, debug=False)
