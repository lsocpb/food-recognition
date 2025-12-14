import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score

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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title('Macierz Pomyłek')
    plt.ylabel('Prawdziwa klasa (Ground Truth)')
    plt.xlabel('Przewidziana klasa (Prediction)')
    plt.show()
    
    return cm

def evaluate_model_pro(model, test_ds, class_names, model_name="Model"):
    print(f"--- Rozpoczynam ewaluację: {model_name} ---")
    
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    df_report = pd.DataFrame(report_dict).transpose()
    
    df_report['support'] = df_report['support'].astype(int)
    
    styled_df = df_report.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score'], vmin=0.5, vmax=1.0)\
                         .format("{:.2%}", subset=['precision', 'recall', 'f1-score'])
    
    print("\nSzczegółowy Raport Klasyfikacji:")
    display(styled_df)

    top3_correct = 0
    for i in range(len(y_true)):
        top3_preds = np.argsort(y_pred_probs[i])[-3:]
        if y_true[i] in top3_preds:
            top3_correct += 1
    top3_acc = top3_correct / len(y_true)
    print(f"\nGlobal Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print(f"Top-3 Accuracy:  {top3_acc:.2%} (Szansa, że poprawny wynik jest w top 3)")

    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(
        cmap='Blues',
        values_format='d',
        ax=axes[0],
        colorbar=False
    )
    axes[0].set_title(f'{model_name} - Macierz Pomyłek (Liczba)')

    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    disp_norm = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
    disp_norm.plot(
        cmap='Reds',
        values_format='.1%',
        ax=axes[1],
        colorbar=False
    )
    axes[1].set_title(f'{model_name} - Macierz Pomyłek (Znormalizowana %)')

    for ax in axes:
        ax.set_xlabel('Przewidziana klasa')
        ax.set_ylabel('Prawdziwa klasa')

    plt.tight_layout()
    plt.show()


    return df_report, cm
