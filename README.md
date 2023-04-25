# Reconhecimento do Alfabeto em Libras
Reconhecimento de Sinais de Letras em Libras utilizando diferentes tecnicas de machine learning.

[Dataset](https://www.kaggle.com/datasets/williansoliveira/libras?resource=download)

## Utilização da Câmera

### Baixando os modelos

Você vai encontrar no [Drive](https://drive.google.com/drive/folders/1FWLEjItsCVNp2cz_t_xlUzyctqFvhMqu?usp=sharing) todos os modelos no formato [modelo][versão]_libras.pt que foram treinados nesse projeto, basta baixar o modelo desejado e colocar no mesmo diretório de cam_reader.py e executar como indicado. As melhores versões dos quatro modelos do projeto estão listados na tabela abaixo.

| Modelo | Acurácia | Link para download | Notebook
|  ---  | ----------- | ---------------- | --------
| ConvNeXt3 | 0.9998   | [convnext3_libras.pt](https://drive.google.com/file/d/1Bv6CR5WcR2eSjWXlWV0IJwWtMU78eZ6F/view?usp=share_link) | [convnext3.ipynb](https://github.com/matheusrbg/Reconhecimento-do-Alfabeto-em-Libras/blob/main/ConvNeXt/convnext3.ipynb) |
| Resnet | 0.9853  | [resnet_libras.pt](https://drive.google.com/file/d/12le1-ssMleU19GBJIa7otkuMNvMor9Zo/view?usp=share_link) | [resnet2.ipynb](https://github.com/matheusrbg/Reconhecimento-do-Alfabeto-em-Libras/blob/main/ResNet/resnet2.ipynb) |
| VGG2  | 1.0   | [VGG19_2_libras.pt](https://drive.google.com/file/d/1-Sh3648G-tRaNsfmO4fzGQGWbIBGhXTn/view?usp=share_link)  | [vgg19bn_2.ipynb](https://github.com/matheusrbg/Reconhecimento-do-Alfabeto-em-Libras/blob/main/VGG/vgg19bn_2.ipynb) |
| GoogLeNet4  | 0.9806  | [googlenet4_libras.pt](https://drive.google.com/file/d/1-5H7juRIZH4uCsAwz2v9N6otOqTwIlGb/view?usp=share_link) | [googlenet4.ipynb](https://github.com/matheusrbg/Reconhecimento-do-Alfabeto-em-Libras/blob/main/GoogLeNet/googlenet4.ipynb) |


### Dependencias

```
pip3 install opencv-python
```

```
pip3 install mediapipe
```

```
pip3 install tqdm
```

```
pip3 install time
```


* torch, torchvision com cuda. Veja [Instalação](https://pytorch.org/get-started/locally/)

### Rodando o Código
```
python cam_reader.py [modelo] [versão]
```


Ex.:
```
python cam_reader.py convnext 3
```
