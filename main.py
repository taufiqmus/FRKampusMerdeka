from cvzone.FaceDetectionModule import FaceDetector
import cv2
import onnxruntime
import numpy as np

from utils import calculate_angle
import json
import time

onnx_model = "resnet50_arcface.onnx"
session = onnxruntime.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

path_json = "data_wajah.json"

cap = cv2.VideoCapture(0)
detector = FaceDetector()

ANGLE_THRES = 50

def pengenalan(emb, path_json):
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

def loop_FR():
    while True:
        success, img = cap.read()
        img_ori = img.copy()
        img_vis, bboxs = detector.findFaces(img)

        if bboxs:
            x, y, w, h = bboxs[0]["bbox"]
            x1, y1, x2, y2 = max(0, x), max(0, y), min(x+w, img_ori.shape[1]), min(y+h, img_ori.shape[0])
            if x > 0 and y > 0:
                cropped_face_img = img_ori[y1:y2, x1:x2]
                crop_vis = cropped_face_img.copy()
                cropped_face_img = cv2.resize(cropped_face_img, (112,112))
                cropped_face_img = cropped_face_img.astype('float32')/255.
                cropped_face_img = cropped_face_img.reshape(1,112,112,3)
                embedding = session.run([output_name], {input_name: cropped_face_img})[0]
                angle, id = pengenalan(embedding, path_json)
                #print(angle)
                if angle <= ANGLE_THRES:
                    print(f"{id} terdeteksi")
                else:
                    print("unknown")

        cv2.imshow("crop_vis", crop_vis)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
try:
    loop_FR()
except:
    pass

cap.release()
cv2.destroyAllWindows()