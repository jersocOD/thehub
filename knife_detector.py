# person_detector.py
import os
import cv2
import torch

def main():
    # 1) Load YOLOv5 model from your local clone + weights
    base    = os.path.dirname(__file__)
    repo    = os.path.join(base, 'yolov5')
    weights = os.path.join(base, 'yolov5s.pt')
    if not os.path.isdir(repo):
        raise FileNotFoundError(f"Missing yolov5 repo at {repo}")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Missing weights at {weights}")
    os.environ['YOLOv5_SKIP_UPDATE'] = '1'
    model = torch.hub.load(repo, 'custom', path=weights, source='local')
    model.conf = 0.5  # only detections ≥50% confidence

    # 2) Open your default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0)")

    # 3) Prepare window & downscale size for faster inference
    display_w = 640
    cv2.namedWindow('PersonDetector', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W = frame.shape[:2]
        # downscale
        small = cv2.resize(frame, (display_w, int(H * display_w / W)))

        # 4) Run inference
        results = model(small)
        dets    = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

        # 5) Filter & draw only class 0 (“person”)
        fx, fy = W / display_w, H / small.shape[0]
        for x1, y1, x2, y2, conf, cls in dets:
            if int(cls) == 0 and conf > model.conf:
                sx1, sy1 = int(x1 * fx), int(y1 * fy)
                sx2, sy2 = int(x2 * fx), int(y2 * fy)
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0,255,0), 2)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, (sx1, sy1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 6) Show
        cv2.imshow('PersonDetector', frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
