# 😊 Detector de Emoções em Tempo Real

Sistema de detecção de emoções facial em tempo real usando DeepFace e OpenCV.

## 📋 Requisitos

- Python 3.8 ou superior
- Webcam funcionando
- Conexão com internet (apenas para o primeiro uso, para baixar modelos)

## 🚀 Instalação

### 1. Clone ou baixe o projeto

```bash
cd Projeto_IA
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv --without-pip
```

### 3. Ative o ambiente virtual

**Windows:**
```bash
venv\Scripts\activate
python -m ensurepip --upgrade
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```
### Instalar deepface, tensorflow e tf_keras"
```bash
pip install deepface opencv-python tensorflow tf-keras
```
## 🎮 Como Usar

### Detecção de Emoções (DeepFace)

Execute o programa principal:

```bash
python deep.py
python emotions_only.py
python recognition_only.py
```

**Recursos:**
- ✅ Detecta 6 emoções: feliz, triste, raiva, surpresa, medo, neutro
- ✅ Detecta gênero: masculino ou feminino
- ✅ Funciona em tempo real
- ✅ Desenha retângulos ao redor dos rostos
- ✅ Mostra a emoção dominante (em verde)
- ✅ Mostra o gênero detectado (em amarelo)
- ✅ Exibe todas as emoções com porcentagens
- ✅ Reconhecimento facial a partir de uma foto modelo em tempo real

**Para sair:** Pressione **Q**


## 📁 Estrutura do Projeto

```
├── deep.py                    # Detector completo: emoções + gênero + reconhecimento facial
├── emotions_only.py           # Detector de emoções e gênero (sem reconhecimento)
├── recognition_only.py        # Reconhecimento facial por embeddings (sem emoções)
├── requirements.txt           # Dependências do projeto
├── known_faces/               # Fotos de pessoas conhecidas (NomePessoa.jpg)
└── README.md                  # Este arquivo
```


## 🤝 Tecnologias Utilizadas

- **DeepFace**: Framework de análise facial
- **OpenCV**: Processamento de imagem e vídeo
- **TensorFlow**: Engine de deep learning
- **Keras**: API de redes neurais

## 📝 Licença

Este projeto usa bibliotecas open-source. Consulte as licenças individuais:
- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV](https://opencv.org/license/)
- [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)

