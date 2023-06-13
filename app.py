from fastai.vision.all import (
    load_learner,
    PILImage,
)
import sys
import os
import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import urllib.request
import pathlib
import mediapipe as mp
import time

# For online hosting
from streamlit_webrtc import webrtc_streamer
import av

# Change PosixPath when os system is window
# st.write(sys.platform)

if sys.platform.startswith('win'):
    pathlib.PosixPath = pathlib.WindowsPath

# make detection mediapipe
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mpPose.Pose()
isLocalhost = False

# python -m streamlit run app.py
FRAME_WINDOW = st.image([])
num_cam=0

# Put st.title on top
st.title('Sitting Posture Estimate')
st.markdown("<hr>", unsafe_allow_html=True)

output_placeholder2 = st.empty()
output_placeholder = st.empty()
pred_placeholder = st.empty()

# Import model
if "Sitting-Poseture-Estimate-model-1-final.pkl" not in os.listdir():
    with st.spinner("Downloading the model from huggingface .."):
        MODEL_URL = "https://huggingface.co/spaces/farrr/Sitting-Poseture-Estimate/resolve/main/Sitting-Poseture-Estimate-model-6-final.pkl"
        urllib.request.urlretrieve(
            MODEL_URL, "Sitting-Poseture-Estimate-model-1-final.pkl")

learn_inf = load_learner('Sitting-Poseture-Estimate-model-1-final.pkl')
SEGMENT_MODEL = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)


def predict(learn, img):
    time.sleep(1)
    pred, pred_idx, pred_prob = learn.predict(img)
    if pred == '00':
        return "00", pred_prob[pred_idx]*100
        # st.error(f"Bad sit with the probability of {pred_prob[pred_idx]*100:.02f}%")
    elif pred == '01':
        return "01", pred_prob[pred_idx]*100
        # st.success(f"Good sit with the probability of {pred_prob[pred_idx]*100:.02f}%")
    elif pred == '02':
        return "02", pred_prob[pred_idx]*100
        # st.warning(f"Unknow with the probability of {pred_prob[pred_idx]*100:.02f}%")

# pipeline : segment people -> replace bg by gray -> predict the image


def segment_out(image, segment_model, bg_color):
    results = segment_model.process(
        cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
    condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.1

    # Generate solid color images for showing the output selfie segmentation mask.

    # replace foreground by mask color
    # fg_image = np.zeros(image.shape, dtype=np.uint8)
    # fg_image[:] = mask_color

    # replace foreground by bg color
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = bg_color

    # Replace background by bg_color
    output_image = np.where(condition, image, bg_image)
    return output_image


def process4webcam(image):
    # Flip horizontally
    image = cv2.flip(image, 1)
    return image


def predict_from_segment(segment_image, learn_inf):
    
    result = predict(learn_inf, segment_image)
    tolerance = 0.01

    if abs(result[1] - 79.0080) < tolerance:
        output_type = "error"
        output_message = "Please stay on camera"
    else:
        if result[0] == "00":
            output_type = "warning"
            output_message = f"Bad sit with the probability of {result[1]:.02f}%"
        elif result[0] == "01":
            output_type = "success"
            output_message = f"Good sit with the probability of {result[1]:.02f}%"

    return output_message, output_type


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    # 24 bit -> 8 bit for each channel
    frame = frame.to_ndarray(format="bgr24")
    frame = process4webcam(frame)
    frame = segment_out(frame,
                        SEGMENT_MODEL,
                        bg_color=(192, 192, 192)
                        )

    output_message, output_type = predict_from_segment(frame, learn_inf)

    if output_type == "success":
        bgr_color = (0, 255, 0)

    elif output_type == "warning":
        bgr_color = (0, 255, 255)

    elif output_type == "error":
        bgr_color = (0, 0, 255)

    frame = cv2.putText(
        frame,
        output_message,
        org=(20, 20),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.5,
        color=bgr_color,
        thickness=1)

    # frame = cv2.resize(frame, (224, 224))
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

def cam_button():
    global num_cam
    st.sidebar.title('Switch camera')
    num_cam = st.sidebar.radio("Select camera", ["Swicth to Camera 1", "Swicth to Camera 2"])
    if num_cam=='Swicth to Camera 1':
       num_cam=0
    else:
       num_cam=1  

def main():
    if isLocalhost:
        cam_button()
        cap = cv2.VideoCapture(num_cam)
        cap.set(3, 224)
        cap.set(4, 224)
        while True:
            ret, frame = cap.read()
            # If camera cannot open
            if not ret:
                output_placeholder.error("Cannot open camera")
                continue
            frame = process4webcam(frame)
            output_placeholder2.image(frame, channels="RGB")
            output = segment_out(frame,
                                 SEGMENT_MODEL,
                                 bg_color=(192, 192, 192)
                                 )
            # Render image
            output_placeholder.image(output, channels="RGB")

            output_message, output_type = predict_from_segment(
                output, learn_inf)
            if output_type == "success":
                pred_placeholder.success(output_message)

            elif output_type == "warning":
                pred_placeholder.warning(output_message)

            elif output_type == "error":
                pred_placeholder.error(output_message)
    else:
        webrtc_streamer(key="sample", video_frame_callback=callback)


if __name__ == '__main__':
    main()

# python -m streamlit run app.py
