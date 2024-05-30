import cv2
import os
from tqdm import tqdm

modelFilee = r"D:\CMPE490-KOD\deploy.prototxt"
configFilee = r"D:\CMPE490-KOD\res10_300x300_ssd_iter_140000.caffemodel"


def load_face_detector():
    modelFile = modelFilee
    configFile = configFilee
    net = cv2.dnn.readNetFromCaffe(modelFile, configFile)
    if cv2.cuda.getCudaEnabledDeviceCount():
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def save_faces_to_video(video_path, output_video_path, net):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (320, 320))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                face_img = frame[y1:y2, x1:x2]
                resized_face = cv2.resize(face_img, (320, 320))
                out.write(resized_face)

    cap.release()
    out.release()


def process_all_videos(input_folder, output_folder, net):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    videos = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi'))]

    for video_file in tqdm(videos):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)
        save_faces_to_video(input_path, output_path, net)


# Usage
net = load_face_detector()
input_folder = r"D:\CMPE490-KOD\dataset\val\manipulated"
output_folder = r"D:\CMPE490-KOD\facefolder\val  \manipulated"
process_all_videos(input_folder, output_folder, net)
