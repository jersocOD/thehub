# main.py
# Weapon‑tracking for Tello—one command at a time + throttled detection

import os

import socket
import time
import cv2
import torch
import numpy as np

# ----------------------------------------------------------------------------
# Tello UDP helper
# ----------------------------------------------------------------------------
class Tello:
    def __init__(self, ip='192.168.10.1', port=8889):
        self.ip, self.port = ip, port
        # commands go out to port 8889, but bind locally to avoid collision
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', 9000))  # use local port 9000
        self.sock.settimeout(5.0)

    def send(self, cmd: str) -> str:
        """Send a command and return its (possibly empty) reply."""
        self.sock.sendto(cmd.encode(), (self.ip, self.port))
        try:
            res, _ = self.sock.recvfrom(1024)
            return res.decode('utf-8', errors='ignore')
        except socket.timeout:
            return ''

    def close(self):
        self.sock.close()


# ----------------------------------------------------------------------------
# WeaponTracker
# ----------------------------------------------------------------------------
class WeaponTracker:
    def __init__(self, tello_ip='192.168.10.1'):
        # Connect & SDK
        self.tello = Tello(ip=tello_ip)
        if self.tello.send('command').strip().lower() != 'ok':
            print("[warn] Couldn't enter SDK mode")
        if self.tello.send('streamon').strip().lower() != 'ok':
            print("[warn] Couldn't start video stream")
        time.sleep(2.0)

        # TAKEOFF 🚁 (auto takeoff)
        if self.tello.send('takeoff').strip().lower() != 'ok':
            print("[warn] takeoff cmd failed")
        time.sleep(5.0)  # wait for climb

        # OpenCV stream (macOS/FFmpeg URI)
        uri = 'udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=500000'
        self.cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("Can't open video stream")
        # drop stale frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cv2.namedWindow('WeaponTracker', cv2.WINDOW_NORMAL)

        # Load YOLOv5 locally
        base    = os.path.dirname(__file__)
        repo    = os.path.join(base, 'yolov5')
        weights = os.path.join(base, 'yolov5s.pt')
        if not os.path.isdir(repo) or not os.path.isfile(weights):
            raise FileNotFoundError("Missing yolov5/ or yolov5s.pt")
        os.environ['YOLOv5_SKIP_UPDATE'] = '1'
        self.model = torch.hub.load(repo, 'custom', path=weights, source='local')

        # Params
        self.center_tol   = 0.10      # ±10% of frame width
        self.size_thresh  = 0.20      # bbox ≥20% width ⇒ close
        self.display_w    = 480       # downscale for detection (was 640)
        self.FPS_DETECT   = 5         # detections per second
        self.CMD_DELAY    = {         # seconds after each command
            'cw': 1.0,
            'ccw':1.0,
            'forward':2.0,
        }

    def _fly_once(self, cmd: str):
        """Send one command and pause for the appropriate delay."""
        print(f">> {cmd}")
        self.tello.send(cmd)
        key = cmd.split()[0]
        time.sleep(self.CMD_DELAY.get(key, 1.0))

    def track_and_approach(self):
        last_detect = 0.0

        while True:
            # Grab & drop stale frames for low latency
            for _ in range(1):
                self.cap.grab()
            ret, frame = self.cap.retrieve()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()
            H, W = frame.shape[:2]

            # only run detection at ~FPS_DETECT
            if now - last_detect >= 1.0/self.FPS_DETECT:
                last_detect = now

                # downscale for speed
                small = cv2.resize(frame, (self.display_w,
                                           int(H*self.display_w/W)))

                # RUN MODEL
                results = self.model(small)
                dets = results.xyxy[0].cpu().numpy()

                # map & filter
                fx, fy = W/self.display_w, H/small.shape[0]
                weapons = []
                for x1,y1,x2,y2,conf,cls in dets:
                    if int(cls) in [0,1,2] and conf>0.5:
                        weapons.append((
                            x1*fx, y1*fy, x2*fx, y2*fy, conf, int(cls)
                        ))

                # decide action
                if not weapons:
                    self._fly_once('cw 30')
                else:
                    x1,y1,x2,y2,conf,cls = max(weapons, key=lambda x:x[4])
                    cx = (x1+x2)/2; bw = (x2-x1)
                    off = cx/W - 0.5

                    if abs(off) > self.center_tol:
                        cmd = 'cw 15' if off>0 else 'ccw 15'
                        self._fly_once(cmd)
                    elif (bw/W) < self.size_thresh:
                        self._fly_once('forward 30')
                    else:
                        print("✓ Centered & close; landing.")
                        break

                # DRAW boxes on full‑res frame
                for x1,y1,x2,y2,_,_ in weapons:
                    cv2.rectangle(frame,
                                  (int(x1),int(y1)),
                                  (int(x2),int(y2)),
                                  (0,0,255),2)

            # SHOW every frame
            cv2.imshow('WeaponTracker', frame)
            if cv2.waitKey(1) == 27:
                print("User aborted.")
                break

        # CLEANUP
        self.cap.release()
        cv2.destroyAllWindows()
        self.tello.send('land')
        self.tello.send('streamoff')
        self.tello.close()


if __name__ == '__main__':
    tracker = WeaponTracker('192.168.10.1')
    tracker.track_and_approach()
