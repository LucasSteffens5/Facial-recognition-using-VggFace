# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:13:06 2020

@author: lucas
"""
## As pastas n podem existir e o diretorio que move ter q estar completo
import shutil
import os

def files_path04(path):
    filenames=[]
    for p, _, files in os.walk(os.path.abspath(path)):
        for file in files:
            filenames.append( os.path.join(p, file))
    return filenames


import pathlib


lucas = os.listdir('C:\\Users\\lucas\\Desktop\\Materias20192\\visaocomputacional\\Comparacaovggs\\lfw_cropp\\')

##lista = list(set(str(path.parent) for path in pathlib.Path(".").glob("**/*jpg")))
#
#lista = [line.replace('lfw_cropp\\', '') for line in lista]

lista2=[]
lista3=[]
for nomefamoso in lucas:
    lista2.append('C:\\Users\\lucas\\Desktop\\Materias20192\\visaocomputacional\\Comparacaovggs\\testemedias\\'+nomefamoso)
    lista3.append('C:\\Users\\lucas\\Desktop\\Materias20192\\visaocomputacional\\Comparacaovggs\\lfw_cropp\\'+nomefamoso)

for i in lista2:
    os.mkdir(i)
    
oldAdress = 'C:\\Users\\lucas\\Desktop\\Materias20192\\visaocomputacional\\Comparacaovggs\\lfw_cropp\\' #pasta origem
newAdress = 'C:\\Users\\lucas\\Desktop\\Materias20192\\visaocomputacional\\Comparacaovggs\\testemedias\\' #pasta destino


t=0
k=0
end =[]
for i in lista3:
    end = files_path04(i)
    for j in end:
        if(t<=7):
            shutil.move(j, lista2[k]) #módulo 'shutil.move()' move os arquivos
            t+=1
    k+=1    
    t=0


