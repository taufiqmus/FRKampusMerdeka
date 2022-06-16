import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import os
from cvzone.FaceDetectionModule import FaceDetector
import shutil
import json

import onnxruntime

from utils import registrasi

onnx_model = "resnet50_arcface.onnx"
session = onnxruntime.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

path_data_wajah = "data_wajah.json"

vid_ = 0 #default kamera pada laptop adalah 0
target_size = (112,112)

class App:
    def __init__(self, window, window_title, video_source=vid_):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.vid = MyVideoCapture(self.video_source)
        self.detector = FaceDetector()
        self.vid_ratio = 1
        
        self.mid_w = int(0.5*self.vid.width)
        self.new_w = int((self.vid.height * (self.vid_ratio)))
        self.new_x = int(self.mid_w - (0.5 * self.new_w))

        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        
        self.num = 0

        self.L2 = tkinter.Label(window, text="NIM:")
        self.L2.pack(anchor=tkinter.CENTER, expand=True,side=tkinter.LEFT)
        self.input = tkinter.Entry(window)
 
        self.input.pack(anchor=tkinter.CENTER, expand=True,side=tkinter.LEFT)

        self.btn_snapshot=tkinter.Button(window, text="daftar", width=50, command=self.snapshot)
        self.btn_snapshot.pack() #anchor=tkinter.CENTER, expand=True
        self.btn_snapshot.config(state='disabled')

        self.btn_del=tkinter.Button(window, text="hapus", width=50, command=self.delete)
        self.btn_del.pack() #anchor=tkinter.CENTER, expand=True
        self.btn_del.config(state='disabled')

        self.cek_terdaftar=tkinter.Button(window, text="cek(?)", width=50, command=self.cek_ids)
        self.cek_terdaftar.pack() #anchor=tkinter.CENTER, expand=True

        self.detection_occured = False
        
        self.delay = 15
        self.update()

        self.window.mainloop()

    def cek_input(self):
        if self.input.get():
            self.btn_snapshot.config(state='normal')
            self.btn_del.config(state='normal')
        else:
            self.btn_snapshot.config(state='disabled')
            self.btn_del.config(state='disabled')
    
    def snapshot(self):
        self.path_snapshot = os.path.join("folder_wajah", self.input.get())
        try:
            os.mkdir(self.path_snapshot)
        except OSError:
            f = os.listdir(self.path_snapshot)
            self.num = len(f)
        else:
            print ("Successfully created the directory %s " % self.path_snapshot)

        if self.detection_occured:
            cv2.imwrite(self.path_snapshot+"/"+self.input.get() + ".jpg", cv2.cvtColor(self.detected_face, cv2.COLOR_RGB2BGR))
            
            cropped_face_img = cv2.resize(self.detected_face, (112,112))
            cropped_face_img = cropped_face_img.astype('float32')/255.
            cropped_face_img = cropped_face_img.reshape(1,112,112,3)

            embedding = session.run([output_name], {input_name: cropped_face_img})
            print(embedding[0])
            registrasi(self.input.get(), embedding[0].tolist(), path_json=path_data_wajah)
            print('saving embedding done.')

    def delete(self):
        try:
            registrasi(self.input.get(), path_json=path_data_wajah, delete=True)
            shutil.rmtree(os.path.join("folder_wajah", self.input.get()))
            print(f'{self.input.get()} telah dihapus.')
        except:
            print(f'{self.input.get()} tidak ada!')
    def cek_ids(self):
        file = open(path_data_wajah)
        data = json.load(file)
        file.close()
        print(data.keys())
    def update(self):
        ret, frame = self.vid.get_frame()
        frame = frame[0:self.vid.height, self.new_x:self.new_x+self.new_w]
        if ret:
            _, bboxs = self.detector.findFaces(frame.copy())
            if bboxs:
                x1, x2, y1, y2 = bboxs[0]["bbox"]
                x, y, w, h = bboxs[0]["bbox"]
                x1, y1, x2, y2 = max(0, x), max(0, y), min(x+w, frame.shape[1]), min(y+h, frame.shape[0])
                self.deteksi_x1 = x1
                self.deteksi_x2 = x2
                self.deteksi_y1 = y1
                self.deteksi_y2 = y2

                f = frame.copy()
                f = f[y1:y2, x1:x2]
                if f.shape[0] > 25 and f.shape[1] > 25: 
                    self.detection_occured = True
                    f = cv2.resize(f, target_size)
                    self.detected_face = f.copy()
                    frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
                else:
                    self.detection_occured = False
        
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.cek_input()   
        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=vid_):
        self.vid = cv2.VideoCapture(video_source)

        ratio_resize = 0.75
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) * ratio_resize)
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * ratio_resize)

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            frame = cv2.resize(frame, (self.width, self.height))
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            pass

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
 
App(tkinter.Tk(), "Registration Tool")
