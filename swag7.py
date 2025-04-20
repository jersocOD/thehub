# main.py
import os
import subprocess
import socket
import threading
import time

import av
import cv2
import torch
from flask import Flask, Response, request, jsonify

# === Audio setup with fallback ===
base_dir   = os.path.dirname(__file__)
sound_path = os.path.join(base_dir, 'THEREISAPERSONHERE2.wav')
if not os.path.isfile(sound_path):
    raise FileNotFoundError(f"Missing sound: {sound_path}")

try:
    import simpleaudio as sa
    wave_obj = sa.WaveObject.from_wave_file(sound_path)
    def play_alert():
        wave_obj.play()
except ImportError:
    def play_alert():
        subprocess.Popen(['afplay', sound_path])

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
Tello_IP           = '192.168.10.1'
COMMAND_PORT       = 8889
VIDEO_PORT         = 11111
LOCAL_COMMAND_PORT = 9000

TARGET_FPS   = 30    # Video refresh rate
DETECT_FPS   = 5     # How often we update person detection
CENTER_TOL   = 0.10  # ±10% of frame width
SIZE_THRESH  = 0.20  # box ≥20% width ⇒ close

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
latest_frame    = None
frame_lock      = threading.Lock()
person_count    = 0
last_sound_time = 0.0
sound_cooldown  = 2.0

# -------------------------------------------------------------------
# Load YOLOv5 locally
# -------------------------------------------------------------------
repo_dir = os.path.join(base_dir, 'yolov5')
weights  = os.path.join(base_dir, 'yolov5s.pt')
if not os.path.isdir(repo_dir):
    raise FileNotFoundError(f"Missing yolov5 repo at {repo_dir}")
if not os.path.isfile(weights):
    raise FileNotFoundError(f"Missing weights at {weights}")

os.environ['YOLOv5_SKIP_UPDATE'] = '1'
model = torch.hub.load(repo_dir, 'custom', path=weights, source='local')
model.conf = 0.5

# -------------------------------------------------------------------
# Tello UDP command socket
# -------------------------------------------------------------------
command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
command_sock.bind(('', LOCAL_COMMAND_PORT))
command_sock.settimeout(1)

def send_command(cmd: str):
    print(f"[SEND] {cmd}")
    try:
        command_sock.sendto(cmd.encode(), (Tello_IP, COMMAND_PORT))
        resp, _ = command_sock.recvfrom(1024)
        print("[RECV]", resp.decode())
    except socket.timeout:
        print("[TIMEOUT]", cmd)
    except Exception as e:
        print("[ERROR]", cmd, e)

# Kick off SDK & video
for c in ('command','streamon'):
    send_command(c)

# -------------------------------------------------------------------
# Flask & video receiver thread
# -------------------------------------------------------------------
app = Flask(__name__)
video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_socket.bind(('', VIDEO_PORT))

def receive_video():
    global latest_frame
    container = av.open(video_socket.makefile('rb'))
    for packet in container.demux():
        for frame in packet.decode():
            img = frame.to_ndarray(format='bgr24')
            with frame_lock:
                latest_frame = img

threading.Thread(target=receive_video, daemon=True).start()

# -------------------------------------------------------------------
# MJPEG generator for Dashboard (raw feed)
# -------------------------------------------------------------------
def generate_mjpeg():
    interval = 1.0 / TARGET_FPS
    while True:
        t0 = time.time()
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is not None:
            disp = cv2.resize(frame, (640,480))
            ret, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY,70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')

        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

# -------------------------------------------------------------------
# MJPEG generator for Person page (detection + steering + audio)
# -------------------------------------------------------------------
def generate_person_mjpeg():
    global person_count, last_sound_time
    interval    = 1.0 / TARGET_FPS
    last_detect = 0.0

    while True:
        t0 = time.time()
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is not None:
            H, W      = frame.shape[:2]
            annotated = frame.copy()
            now       = time.time()

            if now - last_detect >= 1.0/DETECT_FPS:
                last_detect = now

                dw = 640
                dh = int(H * dw / W)
                small = cv2.resize(frame, (dw, dh))

                results = model(small)
                dets    = results.xyxy[0].cpu().numpy()

                people = []
                fx, fy = W/dw, H/dh
                for x1,y1,x2,y2,conf,cls in dets:
                    if int(cls)==0 and conf>model.conf:
                        people.append((x1*fx, y1*fy, x2*fx, y2*fy))

                person_count = len(people)

                if people and (now - last_sound_time) > sound_cooldown:
                    play_alert()
                    last_sound_time = now

                if people:
                    x1,y1,x2,y2 = max(people, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
                    cx  = (x1+x2)/2
                    bw  = (x2-x1)
                    off = cx/W - 0.5
                    if abs(off)>CENTER_TOL:
                        send_command('cw 15' if off>0 else 'ccw 15')
                    elif (bw/W)<SIZE_THRESH:
                        send_command('forward 30')

                for x1,y1,x2,y2 in people:
                    cv2.rectangle(annotated,
                                  (int(x1),int(y1)),
                                  (int(x2),int(y2)), (0,255,0), 2)
                    cv2.putText(annotated, "person",
                                (int(x1),int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0,255,0), 2)

            disp = cv2.resize(annotated, (640,480))
            ret, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY,70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() + b'\r\n')

        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

# -------------------------------------------------------------------
# Routes: Dashboard, feeds, person page, commands
# -------------------------------------------------------------------
@app.route('/')
def dashboard():
    return '''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>🚁 SwarmTag Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f172a;--panel:#1e293b;--accent:#4f46e5;--txt:#f8fafc;
       --green:#10b981;--red:#ef4444;--hudbg:rgba(15,23,42,0.7);}
*{box-sizing:border-box;margin:0;padding:0;}
body{display:grid;grid-template-rows:auto 1fr auto;
     background:var(--bg);color:var(--txt);
     font-family:Inter,sans-serif;height:100vh;overflow:hidden;}
header,footer{background:var(--panel);padding:0.5rem 1rem;
    display:flex;justify-content:space-between;align-items:center;}
header h1{font-size:1.5rem;}header .counter{font-size:1.1rem;font-weight:500;}
main{display:grid;grid-template-columns:2fr 1fr;gap:1rem;
     padding:1rem;height:calc(100vh - 6rem);}
.video{position:relative;background:black;border-radius:8px;overflow:hidden;}
.video img{width:100%;height:100%;object-fit:contain;}
.hud{position:absolute;top:0;left:0;right:0;
     display:flex;justify-content:space-between;
     background:var(--hudbg);padding:0.5rem 1rem;}
.control-container{overflow-y:auto;padding-right:0.5rem;}
.control{display:grid;grid-template-rows:repeat(5,auto);gap:1rem;}
.group{background:var(--panel);border-radius:6px;padding:0.75rem;}
.group h2{font-size:1.1rem;margin-bottom:0.5rem;
          border-bottom:1px solid var(--accent);padding-bottom:0.25rem;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;}
.grid2 button{width:100%;background:var(--accent);border:none;
              border-radius:6px;padding:0.75rem;font-size:1rem;
              font-weight:600;color:var(--txt);cursor:pointer;
              transition:0.2s;}
.grid2 button:hover{background:var(--panel);transform:translateY(-1px);}
.wide{grid-column:span 2;}.takeoff{background:var(--green);}
.land{background:var(--red);}.emergency{background:#b91c1c;
      animation:pulse 1s infinite;}
@keyframes pulse{0%,100%{transform:scale(1);}50%{transform:scale(1.05);}}
</style>
<script>
async function fetchCount(){
  let r=await fetch('/count_people'), j=await r.json();
  document.getElementById('count').innerText = j.count;
}
setInterval(fetchCount,500);
function send(cmd,id){
  document.getElementById(id+'-cmd').value = cmd;
  document.getElementById(id).submit();
}
</script>
</head><body>
<header>
  <h1>🚁 SwarmTag Dashboard</h1>
  <div class="counter">People Detected: <span id="count">0</span></div>
</header>
<main>
  <div class="video">
    <img src="/video_feed" alt="FPV feed">
    <div class="hud">
      <span>Mode: <strong>Manual</strong></span>
      <span>FPS: <strong>--</strong></span>
      <span>Status: <strong>Idle</strong></span>
    </div>
  </div>
  <div class="control-container"><div class="control">
    <div class="group"><h2>Movement</h2><div class="grid2">
      <form id="fmv" action="/command" method="post">
        <input id="fmv-cmd" name="cmd" type="hidden"/>
      </form>
      <button onclick="send('forward 50','fmv')">↑ Forward</button>
      <button onclick="send('back 50','fmv')">↓ Back</button>
      <button onclick="send('left 50','fmv')">← Left</button>
      <button onclick="send('right 50','fmv')">→ Right</button>
    </div></div>
    <div class="group"><h2>Altitude & Rotate</h2><div class="grid2">
      <form id="falt" action="/command" method="post">
        <input id="falt-cmd" name="cmd" type="hidden"/>
      </form>
      <button onclick="send('up 50','falt')">🔼 Up</button>
      <button onclick="send('down 50','falt')">🔽 Down</button>
      <button onclick="send('ccw 45','falt')">⟲ Rotate Left</button>
      <button onclick="send('cw 45','falt')">⟳ Rotate Right</button>
    </div></div>
    <div class="group"><h2>Takeoff / Land</h2><div class="grid2">
      <form id="ftl" action="/takeoff" method="post"></form>
      <form id="fld" action="/land"    method="post"></form>
      <button class="takeoff wide" onclick="document.getElementById('ftl').submit()">🚀 Takeoff</button>
      <button class="land    wide" onclick="document.getElementById('fld').submit()">🛬 Land</button>
    </div></div>
    <div class="group"><h2>Flips</h2><div class="grid2">
      <form id="ffl" action="/command" method="post">
        <input id="ffl-cmd" name="cmd" type="hidden"/>
      </form>
      <button onclick="send('flip f','ffl')">⤴️ Forward</button>
      <button onclick="send('flip b','ffl')">⤵️ Back</button>
      <button onclick="send('flip l','ffl')">↩️ Left</button>
      <button onclick="send('flip r','ffl')">↪️ Right</button>
    </div></div>
    <div class="group"><h2>Utilities</h2><div class="grid2">
      <form id="fut" action="/command" method="post">
        <input id="fut-cmd" name="cmd" type="hidden"/>
      </form>
      <button onclick="send('speed','fut')">🏎️ Speed</button>
      <button onclick="send('battery?','fut')">🔋 Battery</button>
      <button onclick="send('emergency','fut')" class="emergency">🛑 Emergency</button>
      <button onclick="send('command','fut')">🔄 Reconnect</button>
    </div></div>
    <div class="group"><h2>Special</h2><div class="grid2">
      <button onclick="location.href='/person'">👤 Person Detection</button><div></div>
    </div></div>
  </div></div>
</main>
<footer>
  <span>© 2025 SwarmTag Team</span>
  <span>Flask • YOLOv5 • Tello SDK</span>
</footer>
</body></html>'''

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count_people')
def count_people():
    return jsonify(count=person_count)

@app.route('/person')
def person_page():
    return '''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>👤 Person Detection</title>
...
</html>'''  # (Use your existing person-page HTML here)

@app.route('/person_feed')
def person_feed():
    return Response(generate_person_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command', methods=['POST'])
def handle_command():
    cmd = request.form.get('cmd')
    if cmd:
        send_command(cmd)
    return ('',204)

@app.route('/takeoff', methods=['POST'])
def takeoff():
    send_command('takeoff')
    return ('',204)

@app.route('/land', methods=['POST'])
def land():
    send_command('land')
    return ('',204)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=False)
