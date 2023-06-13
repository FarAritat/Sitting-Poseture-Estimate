from fastai.vision.all import (
    load_learner,
    PILImage,
)
import sys
import os
import streamlit as st
import numpy as np
import cv2
import urllib.request
import pathlib
import mediapipe as mp


# Change PosixPath when os system is window
# st.write(sys.platform)

if sys.platform.startswith('win'):
    pathlib.PosixPath = pathlib.WindowsPath

# make detection mediapipe
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mpPose.Pose()

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

def get_image_from_upload():
    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
    if uploaded_file is not None:
        st.image(PILImage.create((uploaded_file)))
        return PILImage.create((uploaded_file))
    return None

def take_a_picture():
    picture = st.camera_input("Take a picture")
    if picture:
        #st.image(picture)
        return PILImage.create((picture)) 
    return None

def main():
    st.sidebar.title('Options')
    datasrc = st.sidebar.radio("Select input source.", ["Uploaded file", "Take a picture"])
    if datasrc == "Uploaded file": 
        frame = get_image_from_upload()
    else:
        frame = take_a_picture()
    result = st.button('Classify')
    if result:
        frame=np.array(frame)
        #st.write(frame)
        output_placeholder2.image(frame, channels="RGB")
        output = segment_out(frame,
                                SEGMENT_MODEL,
                                bg_color=(192, 192, 192)
                                )
        predict(learn_inf, frame)
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

if __name__ == '__main__':
    main()

# python -m streamlit run app.py
