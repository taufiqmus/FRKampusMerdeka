import numpy as np
import json
import math
from sklearn.metrics.pairwise import cosine_similarity

def registrasi(id, embedding=None, path_json=None, delete=False):
    file = open(path_json)
    data = json.load(file)
    file.close()

    file = open("data_nama.json")
    data_nama = json.load(file)
    file.close()

    if delete:
        del data[id]
        with open(path_json, "w") as outfile:
            json.dump(data, outfile)

        del data_nama[id]
        with open("data_nama.json", "w") as outfile:
            json.dump(data_nama, outfile)
    else:
        with open(path_json, "w") as outfile:
            data[id] = embedding
            json.dump(data, outfile)

def calculate_angle(emb1, emb2):
    #print(np.shape(emb1), np.shape(emb2))
    cos_sim=cosine_similarity(emb1.reshape(1,-1),emb2.reshape(1,-1))
    angle = math.acos(cos_sim[0][0])
    angle = math.degrees(angle)
    return angle


#registrasi('a', path_json="data_wajah.json", delete=True)