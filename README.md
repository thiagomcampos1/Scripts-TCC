# 📊 Sistema de Suporte à Decisão para Data Centers com IA Híbrida

> **Projeto de TCC:** Uma abordagem prescritiva para eficiência energética e detecção de falhas em sistemas de climatização de missão crítica.

![Status do Projeto](https://img.shields.io/badge/Status-Concluído-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📝 Sobre o Projeto

O gerenciamento térmico em Data Centers enfrenta desafios críticos de escalabilidade e confiabilidade. Este projeto propõe um **Sistema de Suporte à Decisão (DSS)** que transita da analítica descritiva para a **prescritiva**.

Utilizando uma arquitetura de **Inteligência Artificial Híbrida**, o sistema combina a precisão do *Gradient Boosting*, a explicabilidade da *Regressão Linear* e a detecção de anomalias do *Isolation Forest*. O objetivo é diagnosticar falhas mecânicas, anomalias estruturais e ineficiências energéticas, fornecendo recomendações auditáveis para a equipe de operação.

## 🚀 Funcionalidades Principais

* **Auditoria de Confiabilidade:** Filtragem automática de sensores degradados e equipamentos inconsistentes antes do treinamento (Competição de Algoritmos).
* **Diagnóstico Híbrido:**
    * *Painel Detectivo (GB):* Identifica a importância estatística das variáveis (Causa Raiz).
    * *Painel Prescritivo (LR):* Identifica a direção do ajuste físico (Aumentar/Reduzir).
* **Monitoramento de Integridade:** Detecção de anomalias vetoriais (combinações inválidas de estados) via *Isolation Forest*.
* **Recomendação Prescritiva:** Motor de simulação que sugere o percentual exato de abertura de válvula para recuperar a eficiência ou estabilidade.
* **Dashboard Interativo:** Interface web em Streamlit para visualização em tempo real.

## 🛠️ Arquitetura e Tecnologias

O projeto foi desenvolvido em **Python** seguindo um pipeline rigoroso de Engenharia de Dados:

* **Linguagem:** Python 3.9+
* **Interface:** Streamlit
* **Machine Learning:** Scikit-Learn (GradientBoostingRegressor, LinearRegression, IsolationForest)
* **Visualização:** Plotly Interactive Graphs
* **Processamento de Dados:** Pandas, Numpy
* **Serialização:** Joblib

### Pipeline de Processamento
```mermaid
graph LR
A[Dados Brutos] --> B(01_ETL: Limpeza & Sincronização)
B --> C{02_Auditoria: R² > 0.5?}
C -- Aprovado --> D(03_Treinamento Híbrido)
C -- Reprovado --> X[Descarte / Manutenção]
D --> E[Modelos .joblib]
E --> F(05_Dashboard: Streamlit App)
