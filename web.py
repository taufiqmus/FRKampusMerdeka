import cv2
import streamlit as st
import os
import numpy as np
from PIL import Image

from utils import calculate_angle
from utils import registrasi as regist
import json

import onnxruntime
from cvzone.FaceDetectionModule import FaceDetector
from datetime import datetime
import pandas as pd

ANGLE_THRES = 55

detector = FaceDetector()

onnx_model = "resnet50_arcface.onnx"
session = onnxruntime.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

path_json = "data_wajah.json"

def extract_embedding(img):
    embedding = None
    cropped_face_img = None
    final_box = None
    if img is not None:
        img_ori = img.copy()
        img_vis, bboxs = detector.findFaces(img)
        if bboxs:
            x, y, w, h = bboxs[0]["bbox"]
            x1, y1, x2, y2 = max(0, x), max(0, y), min(x+w, img_ori.shape[1]), min(y+h, img_ori.shape[0])
            final_box = (x1, y1, x2, y2) 
            if x > 0 and y > 0:
                cropped_face_img = img_ori[y1:y2, x1:x2]
                cropped_face_img = cv2.resize(cropped_face_img, (112,112))
                crop_vis = cropped_face_img.copy()
                cropped_face_img = cropped_face_img.astype('float32')/255.
                cropped_face_img = cropped_face_img.reshape(1,112,112,3)
                embedding = session.run([output_name], {input_name: cropped_face_img})[0]
        else:
            pass

    return embedding, crop_vis, final_box 

def pengenalan(emb):
    file = open(path_json)
    data = json.load(file)
    file.close()
    min_angle = 99999
    id = None
    for key in data.keys():
        emb+= 0.000001
        angle = calculate_angle(emb, np.array(data[key]))
        if angle < min_angle:
                min_angle = angle
                id = key
    return min_angle, id    

def save_name(nim, nama):
    file = open("data_nama.json")
    data = json.load(file)
    file.close()
    with open("data_nama.json", "w") as outfile:
        data[nim] = nama
        json.dump(data, outfile)

def get_name(nim):
    file = open("data_nama.json")
    data = json.load(file)
    file.close()
    nama = data[nim]
    return nama

def store_data_to_csv(nama, nim):
    df = pd.read_csv('data_presensi.csv', index_col=0)
    date = datetime.strptime(str(datetime.now().isoformat(' ', 'seconds')), "%Y-%m-%d %H:%M:%S")
    tanggal = date.strftime('%d %b, %Y')
    waktu = date.strftime('%H:%M:%S')
    data = {'Mahasiswa': [nama],
            'NIM': [nim],
            'Tanggal': [tanggal],
            'Waktu': [waktu]
            }
    if len(df) == 0:
        df = pd.DataFrame(data, columns= ['Mahasiswa', 'NIM', 'Tanggal', 'Waktu'])
    else:
        new_df = pd.DataFrame(data)
        df = df.append(new_df, ignore_index=True)
    
    df.to_csv (r'data_presensi.csv', header=True)
    st.write(f"Anda telah melakukan presensi pada {tanggal} pukul {waktu}")

def save_wajah(img, crop, nim):
    path_snapshot = os.path.join("folder_wajah", nim)
    try:
        os.mkdir(path_snapshot)
    except OSError:
        f = os.listdir(path_snapshot)
    else:
        print ("Successfully created the directory %s " % path_snapshot)
    cv2.imwrite(path_snapshot+"/"+nim + "_face.jpg", crop)
    cv2.imwrite(path_snapshot+"/"+nim + "_full.jpg", img)

def presensi(img):
    img_ori = img.copy()
    embedding, crop_wajah, bbox = extract_embedding(img)
    if embedding is not None:
        angle, nim = pengenalan(embedding)
        nama = get_name(nim)

        if angle <= ANGLE_THRES:
            colT1,colT2 = st.columns([3,3])
            with colT1:
                img_ori = cv2.rectangle(img_ori, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (200,170,0), 2)
                crop = st.image([])
                crop.image(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
            with colT2:
                st.write(f"apakah anda {nama} dengan NIM {nim} ?")

                placeholder = st.empty()
                submit_button = placeholder.button('submit')
                    
                if submit_button:
                    store_data_to_csv(nama, nim)
                    placeholder.empty()
        else:
            st.write("ulangi proses, wajah tidak dikenali")
    else:
        st.write("tidak ada wajah yang terdeteksi")

def registrasi(img, nama, nim):
    img_ori = img.copy()
    nama = nama.upper()
    nim = nim.upper()
    embedding, crop_wajah, bbox = extract_embedding(img)
    if embedding is not None:
        if nama=="" or nim=="":
            st.write("isi nama dana NIM terlebih dahulu!")
        else:
            regist(nim, embedding.tolist(), path_json)
            save_name(nim, nama)
            save_wajah(img_ori, crop_wajah, nim)
            st.write(f"{nama} dengan NIM {nim} telah terdaftar.")
    else:
        st.write("ulangi registrasi, tidak ada wajah yang terdeteksi")

st.set_page_config(page_title='Face Recognition')

st.markdown("<h1 style='text-align: center; color: white;'>Smart Presensi</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Selamat datang di sistem presensi Face Recognition Kampus Merdeka!</h3>", unsafe_allow_html=True)
#st.markdown("<h4 style='text-align: center; color: white;'><------ pilih menu registrasi atau presensi</h4>", unsafe_allow_html=True)

st.sidebar.title("Face Recognition")

video_source = st.sidebar.selectbox("Pilihan:", ("Home","Presensi", "Registrasi"))

if video_source in ["Presensi"]:
    img_file_buffer = st.camera_input("")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        presensi(img)


elif video_source in ["Registrasi"]:
    nama = st.sidebar.text_input("masukkan nama: ")
    nim = st.sidebar.text_input("masukkan NIM: ")
    img_file_buffer = st.camera_input("REGISTRASI")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        registrasi(img, nama, nim)




