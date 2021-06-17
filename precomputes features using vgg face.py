# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:26:18 2020

@author: lucas
"""
import pandas as pd
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import os
import numpy as np
from keras.preprocessing import image


model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling=None)# Pode se usar a senet50
	#Faz a predição nas amostras

def ProcuraArquivos(path):  # Percorre todo diretorio e pega o nome dos arquivos
    filenames=[]
    for p, _, files in os.walk(os.path.abspath(path)):
        for file in files:
            filenames.append( os.path.join(p, file))
    return filenames



def GeradorDeVetoresDeCaracteristica(path):
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
def EhUmMatch(known_embedding, candidate_embedding, thresh=0.5):
	#Calcula a distancia do cosseno entre os vetores de caracteristica
    
    
	return cosine(known_embedding, candidate_embedding)


def ExtraiMediaDosVetoresDeCaracteristica(path):
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
diretorioFaces='D:\\SeuCaminho\\lwf\\'
lista = os.listdir(diretorioFaces)


for nomefamoso in lista:
    caminho = diretorioFaces + nomefamoso
    path = ProcuraArquivos(caminho)
    Output = ExtraiMediaDosVetoresDeCaracteristica(path)
    Output = Output.T
    
    famosos.append([nomefamoso, Output])

dfObj = pd.DataFrame(famosos) 

dfObj.to_pickle("./caracteristicas.pkl")