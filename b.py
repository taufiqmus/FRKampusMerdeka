import cv2
import streamlit as st
import os
import numpy as np
from PIL import Image

from time import time

st.set_page_config(page_title='Face Recognition')
st.sidebar.title("Kampus Merdeka")

video_source = st.sidebar.selectbox("Select source:", ("Image", "Webcam", "IP Cam"), key="source_selection")
n_plate = st.sidebar.slider("Set maximum number of plate:", 1,10)

max_total_plate = n_plate
def inference(img=None, max_num_plate=1):
    result = {}
    starttime = time()
    if img is not None:
        formatted_chars, img = extract_plat_info(img_src=img, visualize=True, draw_crop=False, max_num_plate=max_num_plate)
        endtime = time()-starttime
        IMG_VIS = st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        #for stream purpose
        formatted_chars, img = extract_plat_info(img_src=bgr_frame, visualize=True, draw_crop=False, max_num_plate=max_num_plate)
        endtime = time()-starttime
        IMG_VIS = st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 

    st.write("Result: ")
    for i, res in enumerate(formatted_chars):
        one_plate = {}
        one_plate['region'] = res[0]
        one_plate['id'] = res[1]
        result[f'plate-{i}'] = one_plate
    st.write(result)
    st.write(f'Execution time: {endtime:.2f} seconds')


def file_selector(folder_path='video-sample'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file (put your video inside folder "video-sample")', filenames)
    return os.path.join(folder_path, selected_filename)

def select_stream_source(video_source):
    if video_source == "Webcam":
        cam_address = st.sidebar.radio("Select the webcam: ",("0", "1", "2"))
        cam_address = int(cam_address)
    elif video_source == "IP Cam":
        cam_address = st.sidebar.text_input("Input RTSP address and hit enter: ")
    return cam_address

def get_img():
    image_file = st.file_uploader("Upload your image:", accept_multiple_files=False)
    if not image_file:
        return None

    img = Image.open(image_file)
    img = np.array(img)
    return img

    
if video_source in ["Webcam"]:
    img_file_buffer = st.camera_input("Capture License Plate")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        inference(img, max_num_plate=max_total_plate)
        
elif video_source in ["IP Cam"]:
    cam_address = select_stream_source(video_source)
    if "prev_vid_src" in st.session_state:
        tmp_addr = st.session_state["prev_vid_src"]
        cam = cv2.VideoCapture(tmp_addr, cv2.CAP_DSHOW).release()
    
    if video_source == "Webcam":
        cam = cv2.VideoCapture(cam_address, cv2.CAP_DSHOW)
    elif video_source == "IP Cam": 
        cam = cv2.VideoCapture(cam_address)
    else:
        pass
    
    st.session_state["prev_vid_src"] = cam_address
    cam_status = cam.isOpened()
    start = False
    if not cam_status:
        st.write("stream is unavailable, please change the source...")
        
    else:
        start = st.checkbox("AVAILABLE (check to start, uncheck to stop)", key="start")
        FRAME = st.image([])

    if start and cam_status:
        test = st.button("Capture License Plate", key="test", on_click=inference) 
        INF_FRAME = st.image([])   
        while True:
            ret, frame = cam.read()
            if ret and st.session_state.start:
                bgr_frame = frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME.image(rgb_frame)

            else:
                if video_source == "Video":
                    continue
                else:
                    break
    
    cam.release()
else:
    img = get_img()
    
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        inference(img, max_num_plate=max_total_plate)
    else:
        pass


