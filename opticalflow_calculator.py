import cv2
import os
import numpy as np
import multiprocessing
from functools import partial

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = os.path.join(output_dir, os.path.basename(video_path))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out.write(bgr)
        prev_gray = gray

    cap.release()
    out.release()

def main(input_dir, output_dir, num_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4') or f.endswith('.avi')]

    pool = multiprocessing.Pool(num_workers)
    process_func = partial(process_video, output_dir=output_dir)
    pool.map(process_func, video_files)
    pool.close()
    pool.join()

if __name__ == "__main__":
    input_dir = r"D:\CMPE490-KOD\facefolder\val\manipulated\faces"
    output_dir = r"D:\CMPE490-KOD\facefolder\val\manipulated\optical"
    main(input_dir, output_dir)
