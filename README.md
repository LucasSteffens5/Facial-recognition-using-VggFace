# Reconhecimento facial usando o descritor VggFace
Resgatando uma das atividades realizadas durante a graduação (não espere padrão de projetos)

Reconhecimento facial usando o descritor VggFace, juntamente com o extrator de rosto Ultra-Light. A origem de ambos pode ser encontrada nas referências, no entanto, foram feitas modificações no código do extrator de face e um algoritmo foi criado através da saída do descritor de face.

A base de dados utilizada para teste foi <a href="http://vis-www.cs.umass.edu/lfw/">Labeled Faces in the Wild</a>

* O arquivo move_files.py foi usado para mover as fotos de um diretório para outro conforme eram realizados testes.
* O arquivo precomputes features using vgg face.py é usado para percorrer a base de dados com todas as faces  e gerar os vetores de caracteristicas com a vggface, após é gerado e  salvo um vetor média (com bases muito grandes isto terá de ser revisto) dos descritores de cada face juntamente com o nome da face da base e salvo em um .pkl.

3- O arquivo realtime.py carrega o .pkl e usa o extrator de faces citado na referência para extrair a face em tempo real, pela webcam, onde esta extração passa pela vggface e o vetor gerado é classificado pelo vetor com a menor distãncia disponivel no .pkl caso essa distãncia obedeça um limiar minimo.


Caso utilize o ambiente conda o arquivo ambiente.yml contém todo o ambiente configurado para utilizar a implementação.
Basta realizar o seguinte comando no prompt anaconda:

```
conda env create -f reconhecimentoFacialVGGFace.yml
```

Para maiores informações dos códigos aproveitados verifique as referências.


# Hardware Used
CPU: Intel Core i7 7700-HQ

GPU: Nvidia GTX 1050Ti

# References
* [VGG Face Descriptor](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)

* [Real time face recognition with CPU](https://towardsdatascience.com/real-time-face-recognition-with-cpu-983d35cc3ec5)


* [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

* [Keras-vggface](https://github.com/rcmalli/keras-vggface)
