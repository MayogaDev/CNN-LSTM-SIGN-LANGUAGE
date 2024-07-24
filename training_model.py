import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model import get_model
from helpers import get_word_ids, get_sequences_and_labels, create_folder
from constants import *

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
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHIC_PATH, f"confusion_matrix_{model_num}.png"))
    plt.show()

def plot_metrics(metrics, model_nums):
    metrics = np.array(metrics)
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    avg_metrics = np.mean(metrics, axis=0)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_nums))
    width = 0.2
    
    plt.bar(x - 1.5 * width, metrics[:, 0], width, label='Accuracy')
    plt.bar(x - 0.5 * width, metrics[:, 1], width, label='Precision')
    plt.bar(x + 0.5 * width, metrics[:, 2], width, label='Recall')
    plt.bar(x + 1.5 * width, metrics[:, 3], width, label='F1-score')
    
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.title('Performance metric analysis for each fold')
    plt.xticks(np.append(x, x[-1] + 1), list(model_nums) + ['Average'])
    plt.legend()
    
    # Add average bars
    plt.bar(x[-1] + 1 - 1.5 * width, avg_metrics[0], width, color='blue', alpha=0.5)
    plt.bar(x[-1] + 1 - 0.5 * width, avg_metrics[1], width, color='orange', alpha=0.5)
    plt.bar(x[-1] + 1 + 0.5 * width, avg_metrics[2], width, color='green', alpha=0.5)
    plt.bar(x[-1] + 1 + 1.5 * width, avg_metrics[3], width, color='red', alpha=0.5)
    
    plt.savefig(os.path.join(GRAPHIC_PATH, "performance_metrics.png"))
    plt.show()

def training_model_with_plots(model_path, model_num:int, epochs=50):
    word_ids = get_word_ids(KEYPOINTS_PATH)
    sequences, labels = get_sequences_and_labels(word_ids, model_num)
    sequences = pad_sequences(sequences, maxlen=int(model_num), padding='pre', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    model = get_model(int(model_num), len(word_ids))
    history = model.fit(X, y, epochs=epochs, validation_split=0.2)
    model.summary()
    model.save(model_path)
    
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
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, classes=np.unique(labels), model_num=model_num)
    
    # ROC Curve
    y_pred_keras = model.predict(X).ravel()
    fpr, tpr, _ = roc_curve(y.ravel(), y_pred_keras)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(GRAPHIC_PATH, f"roc_curve_{model_num}.png"))
    plt.show()

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    create_folder(GRAPHIC_PATH)
    all_metrics = []
    for model_num in MODEL_NUMS:
        model_path = os.path.join(MODELS_FOLDER_PATH, f"actions_{model_num}.keras")
        metrics = training_model_with_plots(model_path, model_num)
        all_metrics.append(metrics)
    plot_metrics(all_metrics, MODEL_NUMS)
