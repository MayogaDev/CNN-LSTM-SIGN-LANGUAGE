import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import log_loss, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model import get_model
from helpers import get_word_ids, get_sequences_and_labels, create_folder
from keras.callbacks import ReduceLROnPlateau
from constants import *
import time

# Verificar y configurar la GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
else:
    print("No GPU found. Running on CPU.")
    
def plot_confusion_matrix(cm, classes, model_num):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Model {model_num}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHIC_PATH, f"confusion_matrix_{model_num}.png"))
    plt.show()

def plot_metrics(metrics, model_nums):
    accuracy_metrics = []
    precision_metrics = []
    recall_metrics = []
    f1_metrics = []
    mcc_metrics = []
    balanced_accuracy_metrics = []
    kappa_metrics = []
    
    # Extraer solo las métricas que no son listas (exceptuando la especificidad)
    for metric_set in metrics:
        accuracy_metrics.append(metric_set[0])
        precision_metrics.append(metric_set[1])
        recall_metrics.append(metric_set[2])
        f1_metrics.append(metric_set[3])
        mcc_metrics.append(metric_set[5])
        balanced_accuracy_metrics.append(metric_set[6])
        kappa_metrics.append(metric_set[7])
    
    # Convertir a numpy arrays para procesamiento
    accuracy_metrics = np.array(accuracy_metrics)
    precision_metrics = np.array(precision_metrics)
    recall_metrics = np.array(recall_metrics)
    f1_metrics = np.array(f1_metrics)
    mcc_metrics = np.array(mcc_metrics)
    balanced_accuracy_metrics = np.array(balanced_accuracy_metrics)
    kappa_metrics = np.array(kappa_metrics)
    
    avg_metrics = np.array([np.mean(accuracy_metrics), np.mean(precision_metrics), np.mean(recall_metrics), np.mean(f1_metrics), np.mean(mcc_metrics), np.mean(balanced_accuracy_metrics), np.mean(kappa_metrics)])
    
    # Plot de las métricas
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_nums))
    width = 0.2
    
    plt.bar(x - 1.5 * width, accuracy_metrics, width, label='Accuracy')
    plt.bar(x - 0.5 * width, precision_metrics, width, label='Precision')
    plt.bar(x + 0.5 * width, recall_metrics, width, label='Recall')
    plt.bar(x + 1.5 * width, f1_metrics, width, label='F1-score')
    plt.bar(x + 2.5 * width, mcc_metrics, width, label='MCC')
    plt.bar(x + 3.5 * width, balanced_accuracy_metrics, width, label='Balanced Accuracy')
    plt.bar(x + 4.5 * width, kappa_metrics, width, label='Cohen\'s Kappa')

    
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.title('Performance metric analysis for each fold')
    plt.xticks(np.append(x, x[-1] + 1), list(model_nums) + ['Average'])
    plt.legend()
    """
    # Add average bars
    plt.bar(x[-1] + 1 - 1.5 * width, avg_metrics[0], width, color='blue', alpha=0.5)
    plt.bar(x[-1] + 1 - 0.5 * width, avg_metrics[1], width, color='orange', alpha=0.5)
    plt.bar(x[-1] + 1 + 0.5 * width, avg_metrics[2], width, color='green', alpha=0.5)
    plt.bar(x[-1] + 1 + 1.5 * width, avg_metrics[3], width, color='red', alpha=0.5)
    plt.bar(x[-1] + 1 + 2.5 * width, avg_metrics[4], width, color='purple', alpha=0.5)
    plt.bar(x[-1] + 1 + 3.5 * width, avg_metrics[5], width, color='brown', alpha=0.5)
    plt.bar(x[-1] + 1 + 4.5 * width, avg_metrics[6], width, color='pink', alpha=0.5)
    """
    plt.savefig(os.path.join(GRAPHIC_PATH, "performance_metrics.png"))
    plt.show()

def training_model_with_plots(model_path, model_num:int, epochs=100):
    word_ids = get_word_ids(KEYPOINTS_PATH)
    sequences, labels = get_sequences_and_labels(word_ids, model_num)
    sequences = pad_sequences(sequences, maxlen=int(model_num), padding='pre', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    model = get_model(int(model_num), len(word_ids))
    
    # Configura callback para reducir el learning rate si no mejora
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    
    # Registrar tiempo de entrenamiento
    start_time = time.time()
    history = model.fit(X, y, epochs=epochs, validation_split=0.2, verbose=1, callbacks=[reduce_lr])
    end_time = time.time()
    print(f'Training Time: {end_time - start_time:.2f} seconds')
    
    # Plotting loss and accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHIC_PATH, f"training_plots_{model_num}.png"))
    plt.show()
    
    # Predictions
    y_pred_keras = model.predict(X)
    y_pred_classes = np.argmax(y_pred_keras, axis=1)
    y_true = np.argmax(y, axis=1)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')
    # Log Loss
    logloss = log_loss(y, y_pred_keras)
    print(f'Log Loss: {logloss:.2f}')
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred_classes)
    print(f'Matthews Correlation Coefficient: {mcc:.2f}')
    
    # Confusion Matrix and Specificity per class
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Specificity for each class (multiclass scenario)
    specificity_per_class = []
    for i in range(len(np.unique(y_true))):
        tn = cm[i, i]
        fp = np.sum(cm[:, i]) - tn
        specificity = tn / (tn + fp)
        specificity_per_class.append(specificity)

    print(f'Specificity per class: {specificity_per_class}')
    
    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred_classes)
    print(f'Balanced Accuracy: {balanced_acc:.2f}')
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred_classes)
    print(f'Cohen\'s Kappa: {kappa:.2f}')
    
    # Confusion Matrix Plot
    plot_confusion_matrix(cm, classes=np.unique(labels), model_num=model_num)
    
    # ROC Curve for each class (multiclass handling)
    for i in range(y.shape[1]):
        fpr, tpr, _ = roc_curve(y[:, i], y_pred_keras[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{word_ids[i]} ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(GRAPHIC_PATH, f"roc_curve_{model_num}.png"))
    plt.show()
    
    # Save the model and print summary
    model.summary()
    model.save(model_path)
    
    # Return all metrics except specificity_per_class which is a list
    return accuracy, precision, recall, f1, logloss, mcc, balanced_acc, kappa, specificity_per_class

if __name__ == "__main__":
    create_folder(GRAPHIC_PATH)
    all_metrics = []
    all_specificities = []
    for model_num in MODEL_NUMS:
        model_path = os.path.join(MODELS_FOLDER_PATH, f"actions_{model_num}.keras")
        metrics = training_model_with_plots(model_path, model_num)
        all_metrics.append(metrics[:-1])  # Append all metrics except specificity_per_class
        all_specificities.append(metrics[-1])  # Save the specificity_per_class separately

    # Llamada a la función plot_metrics para calcular y mostrar métricas
    plot_metrics(all_metrics, MODEL_NUMS)

    # Graficar las especificidades por clase con colores diferentes
    for i, specificities in enumerate(all_specificities):
        specificities = [0 if np.isnan(val) else val for val in specificities]
        classes = get_word_ids(KEYPOINTS_PATH)

        # Usar colores diferentes para cada barra
        colors = plt.cm.viridis(np.linspace(0, 1, len(specificities)))

        plt.figure(figsize=(12, 6))
        plt.bar(classes, specificities, color=colors)
        plt.title(f'Especificidad por Clase (Modelo {MODEL_NUMS[i]})')
        plt.ylabel('Especificidad')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(GRAPHIC_PATH, f"class_specificity_{model_num}.png"))
        plt.show()
