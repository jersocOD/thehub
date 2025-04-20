import socket
import threading
import av
import numpy as np
import cv2
from flask import Flask, Response, request
import time

TARGET_FPS = 30
interval = 1.0 / TARGET_FPS

# === SETUP ===
Tello_IP = '192.168.10.1'
COMMAND_PORT = 8889
VIDEO_PORT = 11111
LOCAL_COMMAND_PORT = 9000

# === UDP COMMAND SOCKET ===
command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
command_sock.bind(('', LOCAL_COMMAND_PORT))
command_sock.settimeout(1)

def send_command(cmd):
    print(f"[SEND] {cmd}")
    try:
        command_sock.sendto(cmd.encode('utf-8'), (Tello_IP, COMMAND_PORT))
        response, _ = command_sock.recvfrom(1024)
        print(f"[RESPONSE] {response.decode()}")
    except socket.timeout:
        print(f"[TIMEOUT] No response for: {cmd}")
    except Exception as e:
        print(f"[ERROR] Command '{cmd}' failed: {e}")

# === START VIDEO STREAM ===
try:
    send_command("command")
    send_command("streamon")
except Exception as e:
    print(f"[WARN] Drone not responding: {e}")

# === FLASK APP ===
app = Flask(__name__)
video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_socket.bind(('', VIDEO_PORT))

frame_lock = threading.Lock()
latest_frame = None

def receive_video():
    global latest_frame
    container = av.open(video_socket.makefile('rb'))
    for packet in container.demux():
        for frame in packet.decode():
            with frame_lock:
                latest_frame = frame.to_ndarray(format='bgr24')

def generate_mjpeg():
    while True:
        start = time.time()
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
        elapsed = time.time() - start
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
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            body {
                font-family: 'Inter', sans-serif;
                background: var(--bg);
                color: var(--text);
                height: 100vh;
                display: flex;
                flex-direction: column;
                padding: 10px;
                overflow: hidden;
            }
            h1 {
                text-align: center;
                margin: 5px 0;
                font-size: 1.8rem;
            }
            .main-grid {
                display: grid;
                grid-template-rows: 1fr auto;
                height: 100%;
                gap: 10px;
            }
            .video-container {
                background: black;
                border-radius: 8px;
                overflow: hidden;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .video-container img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            .control-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 8px;
            }
            .control-grid button {
                width: 100%;
                height: 100%;
                min-height: 60px;
                font-size: 1.2rem;
                padding: 10px;
            }
            .double-width {
                grid-column: span 2;
            }
            .takeoff {
                background: var(--green);
                border-color: var(--green);
            }
            .land {
                background: var(--red);
                border-color: var(--red);
            }
            .emergency {
                background: #b91c1c;
                border-color: #b91c1c;
                animation: pulse 1s infinite;
            }
            button {
                background: var(--btn-bg);
                color: var(--text);
                border: 2px solid var(--btn-border);
                border-radius: 8px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.2s;
            }
            button:hover {
                background: var(--btn-hover);
                transform: translateY(-2px);
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
                <!-- Movement Controls -->
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="forward 50">
                    <button>↑ Forward</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="left 50">
                    <button>← Left</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="right 50">
                    <button>→ Right</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="back 50">
                    <button>↓ Back</button>
                </form>
                
                <!-- Altitude Controls -->
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="up 50">
                    <button>🔼 Up</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="down 50">
                    <button>🔽 Down</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="ccw 45">
                    <button>⟲ Rotate Left</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="cw 45">
                    <button>⟳ Rotate Right</button>
                </form>
                
                <!-- Action Buttons -->
                <form action="/takeoff" method="post">
                    <button class="takeoff double-width">🚀 Takeoff</button>
                </form>
                <form action="/land" method="post">
                    <button class="land double-width">🛬 Land</button>
                </form>
                
                <!-- Special Functions -->
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="flip l">
                    <button>↩️ Flip Left</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="flip r">
                    <button>↪️ Flip Right</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="flip f">
                    <button>⤴️ Flip Forward</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="flip b">
                    <button>⤵️ Flip Back</button>
                </form>
                
                <!-- Utility Buttons -->
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="speed">
                    <button>🏎️ Speed</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="battery?">
                    <button>🔋 Battery</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="emergency">
                    <button class="emergency">🛑 Emergency Stop</button>
                </form>
                <form action="/command" method="post">
                    <input type="hidden" name="cmd" value="command">
                    <button>Reconnect</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/takeoff', methods=['POST'])
def takeoff():
    send_command("takeoff")
    return '', 204

@app.route('/land', methods=['POST'])
def land():
    send_command("land")
    return '', 204

@app.route('/command', methods=['POST'])
def handle_command():
    cmd = request.form.get('cmd')
    if cmd: send_command(cmd)
    return '', 204

# === MAIN ===
if __name__ == '__main__':
    threading.Thread(target=receive_video, daemon=True).start()
    app.run(host='0.0.0.0', port=5005, debug=False)