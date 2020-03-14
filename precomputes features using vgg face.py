# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:26:18 2020

@author: lucas
"""
import pandas as pd
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine, euclidean
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import os
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from glob import glob
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pathlib
from sklearn.cluster import KMeans
import cv2
import glob
from glob import glob
import feather
import pathlib

from keras.preprocessing import image


model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling=None)# Pode se usar a senet50
	#Faz a predição nas amostras

def files_path04(path):  # Percorre todo diretorio e pega o nome dos arquivos
    filenames=[]
    for p, _, files in os.walk(os.path.abspath(path)):
        for file in files:
            filenames.append( os.path.join(p, file))
    return filenames



def extract_vector(path):
    resnet_feature_list = []

    for im in path:
        #print(im)
        
        img = image.load_img(im, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        resnet_feature = model.predict(x)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())
    ai = np.var(resnet_feature_list)
    return resnet_feature_list, ai








# Verifica se a distancia da entrada com a amostra 
def is_match(known_embedding, candidate_embedding, thresh=0.5):
	#Calcula a distancia do cosseno entre os vetores de caracteristica
    
    
	return cosine(known_embedding, candidate_embedding)


def extract_means_vector(path):
    resnet_feature_list = []
    
    for im in path:
        #print(im)
        
        img = image.load_img(im, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        resnet_feature = model.predict(x)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())
        
    #Cria um vetor com as dimensões para guardar a media
    arr=np.zeros((1,2048),np.float)
    N = len(resnet_feature_list)
    for im in resnet_feature_list:
         imarr=im        
         arr=arr+imarr/N
    
    
    
       
    return arr


famosos=[]

lista = os.listdir('C:\\Users\\lucas\\Desktop\\Materias20192\\visaocomputacional\\Comparacaovggs\\lfw_cropp\\')


for nomefamoso in lista:
    caminho = 'C:\\Users\\lucas\\Desktop\\Materias20192\\visaocomputacional\\Comparacaovggs\\lfw_cropp\\'+nomefamoso
    path = files_path04(caminho)
    Output = extract_means_vector(path)
    Output = Output.T
    
    famosos.append([nomefamoso, Output])

dfObj = pd.DataFrame(famosos) 

dfObj.to_pickle("./caracteristicas.pkl")