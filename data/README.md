# food-recognition
# Food Recognition Project – Repository Template

Ten szablon opisuje strukturę repozytorium, suggested pliki, foldery i krótkie opisy ich roli.

---

## 📁 Struktura katalogów

### Zawartość `data/README.md`

Plik ten opisuje sposób pobierania i przygotowania zbioru danych Food-101 oraz ewentualnych własnych zdjęć.

#### **data/README.md – szablon**

````
# Folder data/

Ten folder NIE przechowuje danych bezpośrednio w repo, aby uniknąć zbyt dużych plików.

## 📥 Jak pobrać dataset Food-101

**Opcja 1 – Pobranie automatyczne (Python):**

```python
import tensorflow_datasets as tfds

dataset, info = tfds.load(
    'food101',
    split=['train', 'validation'],
    as_supervised=True,
    with_info=True
)

train_ds, val_ds = dataset
print("Liczba klas:", info.features['label'].num_classes)
print("Przykładowe klasy:", info.features['label'].names[:10])
````

---

## 📁 Własny dataset

Jeśli dodajecie własne zdjęcia potraw, umieśćcie je w:

```
data/custom/
```

Struktura:

```
custom/
  class_name_1/
  class_name_2/
  ...
```

---

## 🚫 Dlaczego nie trzymamy danych w repo?

* pliki są zbyt duże
* GitHub ma ograniczenia
* łatwiej pobierać on‑the‑fly w Colab/Kaggle

```
```