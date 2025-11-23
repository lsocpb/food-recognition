import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def evaluate_model(model, test_ds, class_names):
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(labels.numpy())
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Macierz Pomyłek')
    plt.ylabel('Prawdziwa klasa (Ground Truth)')
    plt.xlabel('Przewidziana klasa (Prediction)')
    plt.show()
    
    return cm


