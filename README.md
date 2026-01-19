# 🏠 House Price Prediction Project

Benvenuto in questo progetto di Machine Learning.
L'obiettivo è sviluppare un modello predittivo per stimare i prezzi immobiliari partendo da un dataset reale, applicando tecniche di Data Analysis (EDA) e algoritmi di Regressione.

## 🚧 Stato del Progetto
**Work in Progress.**
Migliorando R2 error usando più dati

## 🎯 Obiettivi
- [x] Setup ambiente virtuale e dipendenze
- [x] Acquisizione dataset (Housing Prices)
- [x] Analisi Esplorativa dei Dati (Notebooks)
- [ ] Pulizia dati e Feature Engineering
- [ ] Training del modello (Linear Regression)
- [ ] Valutazione metriche (MSE, R2 Score)

## 🛠️ Tech Stack
- **Python 3.14**
- **Pandas:** Manipolazione dati
- **Matplotlib:** Visualizzazione dati
- **Scikit-Learn:** Modellazione AI
- **Jupyter:** Prototipazione rapida

## 📂 Struttura della Repository
```text
├── data/               # Dataset (ignorato da git per dimensione)
├── notebooks/          # Analisi esplorativa (.ipynb)
├── venv/               # Ambiente virtuale
├── main.py             # Script principale
├── .gitignore          # File esclusi dal versionamento
├── requirements.txt    # Lista dipendenze
└── README.md           # Documentazione
```
## 📂 Dataset
Il progetto utilizza il dataset [Kaggle - Housing Prices](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).

**Istruzioni:**
1. Scarica il file `Housing.csv` dal link sopra.
2. Crea una cartella `data/` nella root del progetto.
3. Inserisci il file csv nella cartella `data/`.

## 🛠️ Installazione
Assicurati di avere Python installato. Clona la repository e installa le dipendenze:

```bash
git clone https://github.com/matteolovato-AI/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```