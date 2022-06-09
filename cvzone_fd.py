from cvzone.FaceDetectionModule import FaceDetector
import cv2
import onnxruntime
import numpy as np

onnx_model = "resnet50_arcface.onnx"
session = onnxruntime.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(2)
detector = FaceDetector()

while True:
    success, img = cap.read()
    img_ori = img.copy()
    img_vis, bboxs = detector.findFaces(img)

    if bboxs:
        print(bboxs[0]["bbox"])
        x, y, w, h = bboxs[0]["bbox"]
        x1, y1, x2, y2 = max(0, x), max(0, y), min(x+w, img_ori.shape[1]), min(y+h, img_ori.shape[0])
        if x > 0 and y > 0:
            cropped_face_img = img_ori[y1:y2, x1:x2]
            crop_vis = cropped_face_img.copy()
            cropped_face_img = cv2.resize(cropped_face_img, (112,112))
            cropped_face_img = cropped_face_img.astype('float32')/255.
            cropped_face_img = cropped_face_img.reshape(1,112,112,3)
            embedding = np.array(session.run([output_name], {input_name: cropped_face_img}))
            # masuk fungsi cek ke data wajah
            print(embedding[0].shape)

    cv2.imshow("crop_vis", crop_vis)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()