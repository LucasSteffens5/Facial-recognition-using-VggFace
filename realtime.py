# -*- coding: utf-8 -*-

import cv2
import numpy as np
import onnx
import pandas as pd
import onnxruntime as ort
from PIL import Image
from onnx_tf.backend import prepare
from scipy.spatial.distance import cosine
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace

#Carrega o modelo
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling=None)# Pode se usar a senet50
	



from keras_vggface.utils import preprocess_input
def EhUmMatch(known_embedding, candidate_embedding, thresh=0.5):
	#Calcula a distancia do cosseno entre os vetores de caracteristica  
    	return cosine(known_embedding, candidate_embedding)

i=0

caracteristicas = pd.read_pickle("./caracteristicas.pkl")
caracteristicas= caracteristicas.rename(columns={0: "Nomes", 1: "Descritor"})
for row in caracteristicas:
   
    descritor = caracteristicas['Descritor']
    pessoa = caracteristicas['Nomes']

def procurapessoa(pessoa,descritor,vetorpredict):
    i=0
    minimo=[]
    
    while( i< len(descritor)-1):
        cos = EhUmMatch(descritor[i],vetorpredict)
        minimo.append(cos)
        
        i+=1
    if(min(minimo)<0.50):    
        indice = minimo.index(min(minimo))
        ret = pessoa[indice]
    else:
        ret = '???'
    
    return ret
    
# Verifica se a distancia da entrada com a amostra 



#Código adpatado das referências 
def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

video_capture = cv2.VideoCapture(0)
#Carrega o modelo de detecção de rostos
onnx_path = 'D:\\SeuCaminho\\ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name
j=0


while True:
    ret, frame = video_capture.read()
    if frame is not None:
        h, w, _ = frame.shape

        # Processa a imagem adquirida
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converte bgr para rgb
        #out=Image.fromarray(img,mode="RGB")
        #out.show()
        img = cv2.resize(img, (640, 480)) # resize
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (80,18,236), 2)
            imagee =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# converte bgr para rgb
            imagee = Image.fromarray(imagee,mode="RGB")
            imagee = imagee.crop((x1, y1, x2, y2))
            imagee= imagee.resize((224,224))
            # Prever VGGFace
            x = image.img_to_array(imagee)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x = model.predict(x)
            x = np.array(x)
            x = x.flatten()
            x = procurapessoa(pessoa,descritor,x)
                    
            #Salva as capturas dos rostos, util para pegar novos rostos e cadastrar na base
            #imagee.save("D:\\SeuCaminho\\img"+str(j)+".jpg")
            j+=1
            cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = x ## Aqui por o nome
            cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        # Pressione q para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Inicializa a WebCam
video_capture.release()
cv2.destroyAllWindows()