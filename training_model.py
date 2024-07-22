import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model import get_model
from helpers import get_word_ids, get_sequences_and_labels, create_folder
from constants import *

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

if __name__ == "__main__":
    create_folder(GRAPHIC_PATH)
    for model_num in MODEL_NUMS:
        model_path = os.path.join(MODELS_FOLDER_PATH, f"actions_{model_num}.keras")
        training_model_with_plots(model_path, model_num)
