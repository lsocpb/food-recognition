# Food Recognition
---

## 📁 Struktura katalogów
```
food-recognition-project/
│
├── data/
│   ├── README.md            # Jak pobrać dane (bez trzymania danych w repo)
│   └── .gitignore           # Ignorujemy duże pliki i dataset
│
├── notebooks/
│   ├── 01_EDA_food101.ipynb
│   ├── 02_Training_EfficientNet.ipynb
│   └── 03_Evaluation.ipynb
│
├── src/
│   ├── preprocessing/
│   │   └── preprocessing.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── models/
│   │   └── model_definitions.py
│   └── utils/
│       └── helpers.py
│
├── models/
│   └── model.h5
│
├── streamlit_app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 📌 README.md (szablon)

### **Food Recognition Project**
System klasyfikacji zdjęć potraw wykorzystujący TensorFlow + Keras oraz dataset Food-101.

---

## 🚀 Uruchomienie

### **1. Klonowanie repozytorium**
```
git clone https://github.com/<user>/food-recognition-.git
cd food-recognition
```

### **2. Virtual environment**
```
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

### **3. Instalacja zależności**
```
pip install -r requirements.txt
```

---

## 🧠 Folder `notebooks/`
Zawiera wszystkie eksperymenty:
- `01_EDA_food101.ipynb` – eksploracja danych
- `02_Training_EfficientNet.ipynb` – trenowanie modeli
- `03_Evaluation.ipynb` – porównanie wyników

---

## ⚙️ Folder `src/`
Moduły projektu gotowe do importu:
- `preprocessing/` – augmentacja, wczytywanie danych
- `training/` – kod treningowy
- `evaluation/` – metryki, confusion matrix
- `models/` – definicje architektur (EfficientNet, MobileNet itd.)
- `utils/` – helpery

---

## 🎮 Streamlit
W folderze `streamlit_app/` znajduje się aplikacja demo:
- możliwość wrzucenia zdjęcia
- predykcja modelu
- wyświetlanie wyników

Uruchomienie:
```
streamlit run streamlit_app/app.py
```

---

