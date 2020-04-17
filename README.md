# Facial-recognition-using-VggFace
Facial recognition using the VggFace descriptor, together with the Ultra-Light face extractor. The origin of both can be found in the references, however modifications were made to the face extractor code and an algorithm was created through the output of the face descriptor.


1- The file move_files.py was used to move the photos from one directory to another as tests were performed.

2- The file precomputes features using vgg face.py is used to browse the database with all faces and generate the feature vectors with the vggface, after which an average vector of the descriptors of each face is generated and saved together with the name from the face of the base and saved to a .pkl.

3- The realtime.py file loads the .pkl and uses the face extractor mentioned in the reference to extract the face in real time, via the webcam, where this extraction goes through vggface and the vector generated is classified by the vector with the shortest distance available in .pkl if that distance meets a minimum threshold.

Sorry my English is still weak, but if you have any questions just send an email to lucassteffensoliveira@hotmail.com



Reconhecimento facial usando o descritor VggFace, juntamente com o extrator de rosto Ultra-Light. A origem de ambos pode ser encontrada nas referências, no entanto, foram feitas modificações no código do extrator de face e um algoritmo foi criado através da saída do descritor de face.

1- O arquivo move_files.py foi usado para mover as fotos de um diretório para outro conforme eram realizados testes.
2- O arquivo precomputes features using vgg face.py é usado para percorrer a base de dados com todas as faces  e gerar os vetores de caracteristicas com a vggface, após é gerado e  salvo um vetor média dos descritores de cada face juntamente com o nome da face da base e salvo em um .pkl.

3- O arquivo realtime.py carrega o .pkl e usa o extrator de faces citado na referência para extrair a face em tempo real, pela webcam, onde esta extração passa pela vggface e o vetor gerado é classificado pelo vetor com a menor distãncia disponivel no .pkl caso essa distãncia obedeça um limiar minimo.





# Hardware Used
CPU: Intel Core i7 7700-HQ

GPU: Nvidia GTX 1050Ti

# References
[VGG Face Descriptor](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)

[Real time face recognition with CPU](https://towardsdatascience.com/real-time-face-recognition-with-cpu-983d35cc3ec5)


[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

[Keras-vggface](https://github.com/rcmalli/keras-vggface)
