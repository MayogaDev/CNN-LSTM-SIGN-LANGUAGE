# Sistema de Reconocimiento de Lenguaje de Señas Peruano (LSP)
## CNN-LSTM Neural Network Implementation

Este proyecto implementa un sistema de reconocimiento de lenguaje de señas peruano utilizando redes neuronales CNN-LSTM y visión por computadora. El sistema es capaz de capturar, procesar y clasificar gestos de lenguaje de señas en tiempo real.

## 🎯 Objetivo del Proyecto

Desarrollar un traductor automático de lenguaje de señas peruano que pueda:
- Capturar gestos mediante cámara web
- Extraer características clave usando MediaPipe
- Clasificar gestos usando modelos LSTM bidireccionales
- Proporcionar retroalimentación por texto y voz

## 🏗️ Arquitectura del Sistema

### Componentes Principales

#### 1. **Pipeline de Datos**
- **Captura de Muestras** (`capture_samples.py`)
  - Captura frames de video de gestos mediante webcam
  - Detecta automáticamente presencia de manos usando MediaPipe
  - Guarda secuencias de frames organizadas por gesto
  - Implementa tiempo de espera para gestos interrumpidos

- **Extracción de Keypoints** (`create_keypoints.py`)
  - Procesa frames capturados usando MediaPipe Holistic
  - Extrae 1,662 puntos clave por frame:
    - Pose: 33 landmarks × 4 coordenadas (x, y, z, visibility)
    - Rostro: 468 landmarks × 3 coordenadas (x, y, z)
    - Mano izquierda: 21 landmarks × 3 coordenadas
    - Mano derecha: 21 landmarks × 3 coordenadas
  - Almacena datos en formato HDF5 para procesamiento eficiente

#### 2. **Arquitectura del Modelo Neural**
- **Modelo Base** (`model.py`)
  ```python
  Bidirectional LSTM (64 units) + BatchNormalization + Dropout(0.4)
  ↓
  LSTM (64 units) + BatchNormalization + Dropout(0.4)
  ↓
  Dense (64 units, ReLU) + BatchNormalization + Dropout(0.4)
  ↓
  Dense (64 units, ReLU)
  ↓
  Dense (output_classes, Softmax)
  ```

- **Características del Modelo:**
  - LSTM bidireccional para capturar patrones temporales en ambas direcciones
  - Regularización L2 (0.001) para prevenir sobreajuste
  - BatchNormalization para estabilizar entrenamiento
  - Dropout (0.4) para mejorar generalización
  - Optimizador Adam con learning rate adaptativo

#### 3. **Sistema de Entrenamiento**
- **Validación Cruzada K-Fold** (`training_model.py`)
  - 5 pliegues para evaluación robusta
  - Métricas comprehensivas: Accuracy, Precision, Recall, F1-Score, MCC, Cohen's Kappa
  - Generación automática de visualizaciones:
    - Matrices de confusión por fold
    - Curvas ROC multiclase
    - Gráficos de entrenamiento (loss/accuracy)
    - Especificidad por clase

- **Optimización de Hiperparámetros:**
  - Integración con Keras Tuner (comentada en código actual)
  - Búsqueda automática de arquitecturas óptimas
  - Reducción automática de learning rate (ReduceLROnPlateau)

#### 4. **Sistema de Evaluación en Tiempo Real**
- **Reconocimiento Adaptativo** (`evaluate_model.py`)
  - Selección dinámica de modelo según longitud de secuencia:
    - Modelo 7: 5-7 frames
    - Modelo 12: 8-12 frames  
    - Modelo 18: 13+ frames
  - Umbral de confianza configurable (default: 0.6)
  - Captura automática de frames durante reconocimiento
  - Retroalimentación por texto y voz usando gTTS

## 📊 Dataset y Clases

### Gestos Reconocidos (18 clases):
```
- acceso           - barra de herramientas    - borde exterior
- cancelar         - click derecho           - comando
- computacion      - contrasenia             - copiar texto
- correo electronico - cortar               - dibujar tabla
- disco duro       - escape                 - pantalla
- regresar         - seleccionar            - software
```

### Estructura de Datos:
```
data/
├── keypoints/          # Archivos HDF5 con keypoints procesados
│   ├── acceso.h5
│   ├── comando.h5
│   └── ...
frame_actions/          # Frames de video originales
├── acceso/
│   ├── sample_240828215753129819/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── ...
└── ...
```

## 🔬 Resultados y Métricas

### Evaluación del Modelo:
El sistema genera métricas detalladas incluyendo:

- **Matrices de Confusión** por cada fold de validación cruzada
- **Curvas ROC** multiclase con AUC scores
- **Métricas de Clasificación:**
  - Accuracy, Precision, Recall, F1-Score
  - Matthews Correlation Coefficient (MCC)
  - Balanced Accuracy, Cohen's Kappa
  - Especificidad por clase

### Visualizaciones Generadas:
```
graphics/
├── confusion_matrix_fold_1.png
├── roc_curve_matrix_fold_1_model_18.png
├── training_plots_fold_1.png
├── class_specificity_fold_1.png
├── performance_metrics.png
└── metrics_table.png
```

## 💻 Instalación y Uso

### Requisitos:
```bash
pip install -r requirements.txt
```

### Dependencias Principales:
- TensorFlow 2.10.1
- MediaPipe 0.10.11
- OpenCV 4.9.0.80
- Keras 2.10.0
- NumPy 1.26.4
- Pandas 2.2.2

### Flujo de Trabajo:

1. **Captura de Datos:**
   ```bash
   python capture_samples.py  # Modificar word_name en el script
   ```

2. **Procesamiento de Keypoints:**
   ```bash
   python create_keypoints.py  # Configurar word_ids para procesar
   ```

3. **Entrenamiento del Modelo:**
   ```bash
   python training_model.py
   ```

4. **Evaluación en Tiempo Real:**
   ```bash
   python evaluate_model.py
   ```

## 🛠️ Características Técnicas

### Optimizaciones Implementadas:
- **GPU Acceleration:** Configuración automática de memoria de GPU
- **Data Augmentation:** Padding adaptativo de secuencias
- **Regularización:** L2, Dropout, BatchNormalization
- **Early Stopping:** Reducción de learning rate basada en val_loss
- **Memory Efficiency:** Uso de HDF5 para almacenamiento eficiente

### Configuraciones Flexibles:
- `constants.py` centraliza parámetros importantes
- Modelos adaptativos según longitud de secuencia
- Umbrales de confianza ajustables
- Soporte para múltiples modelos simultáneos

## 🔍 Análisis del Código

### Patrones de Diseño Utilizados:
- **Separation of Concerns:** Módulos especializados para cada función
- **Factory Pattern:** Creación dinámica de modelos según parámetros
- **Strategy Pattern:** Selección de modelo basada en longitud de secuencia
- **Pipeline Pattern:** Flujo estructurado de procesamiento de datos

### Calidad del Código:
- Documentación consistente en español
- Funciones modulares y reutilizables
- Manejo de errores en operaciones críticas
- Configuración centralizada en constantes

## 🚀 Posibles Mejoras

### Técnicas:
1. **Aumentación de Datos:** Rotaciones, escalado, ruido
2. **Arquitecturas Avanzadas:** Transformer, Attention Mechanisms
3. **Optimización de Modelos:** Quantization, Pruning
4. **Multi-modal Learning:** Integración de audio/contexto

### Funcionales:
1. **Interface Gráfica:** GUI para facilidad de uso
2. **Base de Datos:** Sistema de gestión de usuarios y sesiones
3. **Cloud Deployment:** API REST para integración
4. **Mobile Support:** Aplicación móvil nativa

## 📈 Resultados del Análisis

Este proyecto demuestra una implementación sólida y bien estructurada de un sistema de reconocimiento de lenguaje de señas. La arquitectura modular, el uso de técnicas de deep learning modernas, y la evaluación rigurosa mediante validación cruzada indican un desarrollo profesional con potencial para aplicaciones reales.

La combinación de MediaPipe para extracción de características y LSTM bidireccional para modelado temporal representa una aproximación efectiva para el problema de reconocimiento de gestos secuenciales.