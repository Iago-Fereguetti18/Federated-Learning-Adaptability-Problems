# Federated-Learning-Adaptability-Problems

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## Sobre o Projeto
Este projeto é fruto de uma pesquisa de Iniciação Científica em Ciência da Computação (PUC Minas). O objetivo principal é simular e solucionar os maiores desafios do **Aprendizado Federado Personalizado (PFL)** em ambientes de Internet das Coisas (IoT), com foco em dispositivos de borda com recursos limitados.

O simulador aborda de frente dois problemas clássicos:
1. **Heterogeneidade Espacial (Non-IID / Label Skew):** Resolvido através da implementação do framework **PerFit** (Federated Transfer Learning), que congela as camadas base da rede neural para economizar bateria e personaliza apenas as camadas finais.
2. **Heterogeneidade Temporal (Concept Drift):** Resolvido através da integração do algoritmo estatístico **CUSUM**. Quando uma mudança brusca de comportamento é detectada, o sistema aciona um **Descongelamento Dinâmico** da rede neural, permitindo rápida adaptação à nova realidade sem intervenção manual.

## Principais Funcionalidades
* **Dashboard Interativo:** Interface construída em Streamlit para visualização em tempo real do treinamento, permitindo acompanhar a acurácia Global vs. Local a cada rodada.
* **Mecanismo de Defesa Híbrido:** O modelo opera no modo econômico (PerFit) na maior parte do tempo, gastando processamento extra (Full Fine-Tuning) apenas quando o alarme de *drift* do CUSUM dispara.
* **Cenários de Teste Extremos:**
  * *Cenário Espacial:* Simula a diferença de rotina "de fábrica" entre usuários usando *Label Skew*.
  * *Cenário Temporal:* Simula um terremoto nos dados (*Feature Permutation*) para testar a resiliência do modelo a mudanças repentinas.
* **Múltiplas Arquiteturas:** Suporte para Redes Multilayer Perceptron (MLP) e Redes Convolucionais (CNN) de 1D.

## Tecnologias e Bibliotecas
* **Linguagem:** Python
* **IA & Deep Learning:** PyTorch, NumPy, Pandas
* **Visualização:** Streamlit, Altair

## Instruções
### No Windows
* python -m venv .venv
* .venv\Scripts\activate

### No Linux/Mac
* python3 -m venv .venv
* source .venv/bin/activate

### Instale as dependências:
* pip install -r requirements.txt

## 📂 Estrutura do Repositório
```text
├── data/                  # Base de dados (ex: UCI HAR modificado)
├── src/
│   ├── client.py          # Lógica do Cliente Federado (PerFit + CUSUM)
│   ├── server.py          # Lógica do Servidor (Agregação FedAvg)
│   ├── models.py          # Arquiteturas de Redes Neurais (MLP e CNN)
│   ├── data_loader.py     # Processamento e injeção de cenários de teste
│   └── simulation.py      # Orquestrador do loop de treinamento federado
├── dashboard.py           # Aplicação principal (Interface Streamlit)
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação

#Execute o painel interativo:
streamlit run dashboard.py
