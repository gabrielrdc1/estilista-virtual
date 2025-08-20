# 10. Justificativa das Escolhas Técnicas

### Por que MobileNetV2?
- MobileNetV2 foi escolhida por ser leve, eficiente e apresentar bom desempenho em tarefas de classificação de imagens, mesmo com datasets menores, o que é comum em projetos acadêmicos.
- Permite transfer learning, acelerando o treinamento e melhorando a acurácia.

### Por que KMeans para cor dominante?
- KMeans é simples, rápido e eficiente para clusterização de cores em imagens.
- Permite identificar as cores mais representativas sem necessidade de redes neurais adicionais.

### Por que Flask?
- Flask é um framework web minimalista, fácil de integrar com modelos Python e ideal para protótipos e MVPs.

### Por que Chart.js no dashboard?
- Chart.js é leve, fácil de integrar com Flask/Jinja2 e suficiente para visualizações interativas básicas.

### Outras escolhas:
- Uso de JSONL para log: fácil de manipular, permite registro incremental e leitura eficiente.

# 11. Abordagens Testadas e Problemas Encontrados

## CNN
- Testamos ResNet50 e VGG16, mas apresentaram overfitting e maior tempo de treinamento, sem ganho significativo de acurácia em relação à MobileNetV2.
- Modelos mais pesados dificultaram o deploy local e aumentaram o tempo de resposta da API.

## Identificação de Cor
- Testamos histogramas de cor e quantização simples, mas os resultados eram menos robustos para imagens com iluminação variada.
- Algoritmos baseados em deep learning para cor foram descartados por complexidade e tempo de processamento.

## Backend
- Consideramos FastAPI, mas Flask atendeu melhor pela simplicidade e integração com templates HTML.

## Dashboard
- Testamos plotly, mas Chart.js foi mais simples para integração e customização rápida.

## Outros desafios
- Dificuldade em encontrar datasets balanceados para todas as classes.
- Ajuste de hiperparâmetros do KMeans para evitar clusters irrelevantes.
- Problemas de performance ao processar imagens muito grandes (resolvido com resize prévio).
# Detalhes Técnicos — Estilista Virtual CNN

## Visão Geral

O Estilista Virtual CNN é um sistema de classificação de roupas e identificação de cor dominante em imagens, utilizando uma CNN baseada em MobileNetV2 e algoritmos de clusterização de cor. O backend é em Python com Flask, e a interface web permite upload, visualização de resultados e dashboard de análises.

---

## 1. Arquitetura da CNN

- **Base:** MobileNetV2 (pré-treinada no ImageNet, fine-tuning para o domínio de roupas)
- **Camadas adicionais:**
  - GlobalAveragePooling2D
  - Dense (ReLU)
  - Dropout (opcional)
  - Dense (softmax para classificação)
- **Framework:** TensorFlow/Keras
- **Saídas:** 6 classes (blusa/calça/tênis, masculino/feminino)

### Exemplo de código (resumido):
```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(6, activation='softmax')(x)
model = Model(inputs=base.input, outputs=output)
```

---

## 2. Pipeline de Dados

- **Organização:**
  - `data/train/<classe>/` e `data/val/<classe>/` (imagens por categoria)
- **Pré-processamento:**
  - Redimensionamento para 224x224
  - Normalização dos pixels
  - Augmentação: flips, rotações, zoom, brightness
- **Divisão:**
  - Treino/validação (pasta separada)

---

## 3. Treinamento

- **Script:** `modelo/treino_cnn.py`
- **Callbacks:** EarlyStopping, ModelCheckpoint
- **Métricas:** accuracy, loss
- **Salvamento:** modelo final em `modelo/cnn_model_mobilenetv2.h5`

---

## 4. Identificação de Cor Dominante

- **Algoritmo:** KMeans (scikit-learn)
- **Processo:**
  - Crop da peça detectada
  - Redução de resolução para acelerar
  - KMeans com k=3 ou k=5
  - Conversão do centroide para nome de cor (regra própria)
- **Exemplo:**
```python
from sklearn.cluster import KMeans
import numpy as np

pixels = crop_img.reshape(-1, 3)
kmeans = KMeans(n_clusters=3).fit(pixels)
main_color = kmeans.cluster_centers_[0]
```

---

## 5. Backend Flask

- **Rotas principais:**
  - `/` (upload e resultado)
  - `/dashboard` (dashboard de análises)
  - `/login`, `/register` (autenticação)
- **Integração:**
  - Carregamento do modelo na inicialização
  - Processamento de imagem e resposta JSON/HTML

---

## 6. Dashboard

- **Tecnologia:** Flask + Jinja2 + Chart.js
- **Funcionalidades:**
  - Histórico de análises (log em `data/analysis_log.jsonl`)
  - Estatísticas de uso
  - Gráficos de categorias e cores mais frequentes
  - Visualização de imagens analisadas

---

## 7. Dependências Principais

- Python >= 3.8
- Flask
- TensorFlow/Keras
- scikit-learn
- numpy, pandas
- matplotlib, Pillow

---

## 8. Fluxo de Uso

1. Usuário faz upload de imagem
2. Sistema detecta e classifica peça (CNN)
3. Extrai cor dominante (KMeans)
4. Exibe resultado e salva no log
5. Dashboard mostra estatísticas e histórico

---

## 9. Observações

- O sistema é para fins acadêmicos/demonstração.
- Para produção, recomenda-se revisão de segurança, performance e escalabilidade.
