# Estilista Virtual CNN

Este projeto utiliza uma rede neural convolucional (CNN) baseada em MobileNetV2 para classificar peças de roupas em imagens, além de identificar a cor dominante de cada peça. O sistema foi desenvolvido em Python com Flask para servir uma interface web.

## Principais Funcionalidades

- **Classificação de Roupas:**  
	Utiliza uma CNN treinada para identificar categorias como blusa, calça e tênis, separando por sexo (masculino/feminino).

- **Identificação de Cor Dominante:**  
	Algoritmo de KMeans para extrair a cor predominante da peça classificada, mapeando para nomes comuns (ex: Azul, Vermelho, Marrom, etc).

- **Interface Web:**  
	Permite upload de imagens, visualização dos resultados de classificação e cor, além de histórico de análises.

- **Dashboard:**  
	O sistema conta com um dashboard interativo onde o usuário pode visualizar o histórico de análises realizadas, estatísticas de uso, categorias mais frequentes e cores predominantes detectadas. O dashboard facilita o acompanhamento dos resultados e a navegação pelas análises anteriores.

## Estrutura do Projeto

- `app.py` — Aplicação Flask principal.
- `modelo/treino_cnn.py` — Script de treinamento da CNN com MobileNetV2.
- `modelo/cnn_model_mobilenetv2.h5` — Modelo treinado salvo.
- `static/` — Imagens temporárias e uploads.
- `templates/` — Templates HTML para a interface.
- `data/` — Dados de treino, validação e logs.
- `auth/` — Autenticação de usuários.

## Como Executar

1. **Instale as dependências:**
	 ```
	 pip install -r requirements.txt
	 ```

2. **Execute o servidor Flask:**
	 ```
	 python app.py
	 ```

3. **Acesse a interface:**  
	 Abra [http://localhost:5000](http://localhost:5000) no navegador.

## Treinamento do Modelo

O treinamento é feito via [`modelo/treino_cnn.py`](modelo/treino_cnn.py), utilizando imagens organizadas em pastas por classe dentro de `data/train` e `data/val`. O modelo final é salvo em [`modelo/cnn_model_mobilenetv2.h5`](modelo/cnn_model_mobilenetv2.h5).

## Identificação de Cor

A função de extração de cor dominante está em [`app.py`](app.py), utilizando KMeans para clusterização dos pixels e regras para nomear as cores.

## Exemplo de Uso

1. Faça upload de uma imagem de roupa.
2. O sistema retorna a categoria (ex: "blusa_feminino") e a cor dominante (ex: "Azul").
3. Os resultados são exibidos na interface web.

## Créditos

Desenvolvido por Gabriel Cristo.

---

> **Observação:** Este projeto é para fins acadêmicos/demonstração. Para uso em produção, recomenda-se revisar questões de segurança, performance e escalabilidade.
