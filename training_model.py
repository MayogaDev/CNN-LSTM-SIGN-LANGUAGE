import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import log_loss, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold  # Para validación cruzada
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model import get_model
from helpers import get_word_ids, get_sequences_and_labels, create_folder
from keras.callbacks import ReduceLROnPlateau
from constants import *
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LeakyReLU
from keras.regularizers import l2
import time
import kerastuner as kt

# Verificar y configurar la GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
else:
    print("No GPU found. Running on CPU.")
    
def plot_confusion_matrix(cm, classes, model_num,fold):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix Fold {fold + 1}')
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
    plt.savefig(os.path.join(GRAPHIC_PATH, f"confusion_matrix_fold_{fold + 1}.png"))
    #plt.show()

def save_metrics_table(metrics, save_path):
    """
    Genera y guarda una tabla de métricas en formato de imagen.
    """
    columns = ["Fold", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", 
               "MCC", "Balanced Accuracy (%)", "Cohen's Kappa"]
    
    fold_numbers = [str(i+1) for i in range(len(metrics))] + ["Promedio"]
    
    metrics = np.round(np.array(metrics) * 100, 3)
    avg_metrics = np.round(np.mean(metrics, axis=0), 3)
    
    data = np.vstack([metrics, avg_metrics])
    
    df = pd.DataFrame(data, columns=columns[1:])
    df.insert(0, columns[0], fold_numbers)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

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
    plt.savefig(os.path.join(GRAPHIC_PATH, "performance_metrics.png"))
    #plt.show()

# Nueva función para la optimización de hiperparámetros con Keras Tuner
def build_model_with_hp_tuning(hp):
    max_length_frames = hp.Int('max_length_frames', min_value=50, max_value=200, step=50)
    output_length = 14
    
    model = tf.keras.Sequential()
    model.add(Bidirectional(LSTM(units=hp.Int('lstm_units', min_value=64, max_value=128, step=32),
                                 return_sequences=True, activation='tanh',
                                 input_shape=(max_length_frames, LENGTH_KEYPOINTS),
                                 kernel_regularizer=l2(0.001))))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(LSTM(units=hp.Int('lstm_units_2', min_value=64, max_value=128, step=32),
                   return_sequences=False, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(Dense(units=hp.Int('dense_units', min_value=64, max_value=128, step=32), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(Dense(output_length, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG')),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def plot_roc_curves(y_true, y_pred, num_classes, model_num, fold_num):
    """
    Graficar la Curva ROC para cada clase en una matriz de 5x4 y guardarlo por fold.
    """
    # Convertir las etiquetas a formato binarizado
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    
    # Crear una matriz de subgráficos de 5x4
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    axes = axes.ravel()
    
    # Calcular la curva ROC y AUC para cada clase
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Graficar la curva ROC en el subgráfico correspondiente
        axes[i].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        axes[i].plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal (sin habilidad)
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'Class {i+1}')
        axes[i].legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHIC_PATH, f"roc_curve_matrix_fold_{fold_num}_model_{model_num}.png"))
    #plt.show()

def training_model_with_plots(model_path, model_num:int, epochs=100):
    word_ids = get_word_ids(KEYPOINTS_PATH)
    sequences, labels = get_sequences_and_labels(word_ids, model_num)
    sequences = pad_sequences(sequences, maxlen=int(model_num), padding='pre', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    
    # Configura la validación cruzada (KFold para multilabel)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []
    all_specificities = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\nTraining Fold {fold + 1}...")
        
        # Dividir datos en entrenamiento y validación
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Optimización de hiperparámetros con Keras Tuner
        # tuner = kt.Hyperband(build_model_with_hp_tuning, objective='val_accuracy', max_epochs=10, factor=3, directory='my_dir', project_name='text_classification')
        # tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

        # best_model = tuner.get_best_models(num_models=1)[0]
        model = get_model(int(model_num), len(word_ids))
    
        # Configura callback para reducir el learning rate si no mejora
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

        # Entrenar el mejor modelo
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=1, callbacks=[reduce_lr])

        # Predicciones
        y_pred_keras = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred_keras, axis=1)
        y_true = np.argmax(y_val, axis=1)

        # Métricas
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='weighted')
        recall = recall_score(y_true, y_pred_classes, average='weighted')
        f1 = f1_score(y_true, y_pred_classes, average='weighted')
        logloss = log_loss(y_val, y_pred_keras)
        mcc = matthews_corrcoef(y_true, y_pred_classes)
        balanced_acc = balanced_accuracy_score(y_true, y_pred_classes)
        kappa = cohen_kappa_score(y_true, y_pred_classes)

        # Imprimir métricas
        print(f"Fold {fold + 1} Metrics:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")
        print(f"Log Loss: {logloss:.2f}")
        print(f"MCC: {mcc:.2f}")
        print(f"Balanced Accuracy: {balanced_acc:.2f}")
        print(f"Cohen's Kappa: {kappa:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plot_confusion_matrix(cm, classes=np.unique(labels), model_num=model_num,fold=fold)

        # Calcular especificidad
        specificity_per_class = []
        for i in range(len(np.unique(y_true))):
            tn = cm[i, i]
            fp = np.sum(cm[:, i]) - tn
            specificity = tn / (tn + fp)
            specificity_per_class.append(specificity)
        print(f"Specificity per class: {specificity_per_class}")

        # Guardar métricas
        all_metrics.append([accuracy, precision, recall, f1, mcc, balanced_acc, kappa])
        all_specificities.append(specificity_per_class)

        # Graficar las métricas de pérdida y precisión
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss Over Epochs (Fold {fold + 1})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Accuracy Over Epochs (Fold {fold + 1})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHIC_PATH, f"training_plots_fold_{fold + 1}.png"))
        #plt.show()
        
        plot_roc_curves(y_true, y_pred_keras, num_classes=len(np.unique(labels)), model_num=model_num, fold_num=fold + 1)


    # Save the model and print summary
    model.summary()
    model.save(model_path)
    
    # Cálculo de promedios
    avg_metrics = np.mean(np.array(all_metrics), axis=0)
    print("\nAverage Metrics Across All Folds:")
    print(f"Average Accuracy: {avg_metrics[0]:.2f}")
    print(f"Average Precision: {avg_metrics[1]:.2f}")
    print(f"Average Recall: {avg_metrics[2]:.2f}")
    print(f"Average F1-score: {avg_metrics[3]:.2f}")
    print(f"Average MCC: {avg_metrics[4]:.2f}")
    print(f"Average Balanced Accuracy: {avg_metrics[5]:.2f}")
    print(f"Average Cohen's Kappa: {avg_metrics[6]:.2f}")
    
    # Guardar tabla de métricas como imagen
    save_metrics_table(all_metrics, os.path.join(GRAPHIC_PATH, "metrics_table.png"))

    # Llamada a la función plot_metrics para calcular y mostrar métricas
    # plot_metrics(all_metrics, range(1, 6))
    
    # Graficar las especificidades por clase con colores diferentes
    for i, specificities in enumerate(all_specificities):
        specificities = [0 if np.isnan(val) else val for val in specificities]
        classes = get_word_ids(KEYPOINTS_PATH)

        colors = plt.cm.viridis(np.linspace(0, 1, len(specificities)))

        plt.figure(figsize=(12, 6))
        plt.bar(classes, specificities, color=colors)
        plt.title(f'Especificidad por Clase (Fold {i + 1})')
        plt.ylabel('Especificidad')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(GRAPHIC_PATH, f"class_specificity_fold_{i + 1}.png"))
        #plt.show()
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
"""
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
"""