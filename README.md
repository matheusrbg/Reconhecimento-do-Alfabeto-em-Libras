# Reconhecimento-do-Alfabeto-em-Libras
Reconhecimento de Sinais de Letras em Libras utilizando diferentes tecnicas de machine learning

[Dataset](https://www.kaggle.com/datasets/williansoliveira/libras?resource=download)

[Drive](https://drive.google.com/drive/folders/1FWLEjItsCVNp2cz_t_xlUzyctqFvhMqu?usp=sharing)

## Utilização da Câmera

### Baixando os modelos

Você vai encontrar no [Drive](https://drive.google.com/drive/folders/1FWLEjItsCVNp2cz_t_xlUzyctqFvhMqu?usp=sharing) os modelos no formato [modelo][versão]_libras.pt, basta baixar esse aplicativo e colocar no mesmo diretório de cam_reader.py e executar como indicado

### Dependencias

* pip install opencv-python

* pip install mediapipe

* pip install time

* torch, torchvision com cuda. Veja [Instalação](https://pytorch.org/get-started/locally/)

### Rodando o Código
  python cam_reader.py [modelo] [versão]

  Ex.: python cam_reader.py convnext 3
