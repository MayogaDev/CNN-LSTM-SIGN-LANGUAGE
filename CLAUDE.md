# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive sign language recognition system using CNN-LSTM neural networks to classify Peruvian Sign Language (LSP) gestures through computer vision and MediaPipe pose estimation. The system implements a complete pipeline from data capture to real-time gesture recognition with text-to-speech feedback, designed as an automatic translator for Peruvian sign language.

**Core Capabilities:**
- Real-time gesture capture via webcam with automatic hand detection
- Advanced feature extraction using MediaPipe Holistic (1,662 keypoints per frame)
- Bidirectional LSTM neural networks for temporal pattern recognition
- Multi-model adaptive selection based on gesture sequence length
- Comprehensive evaluation with K-fold cross-validation and detailed metrics
- Text-to-speech feedback for accessibility

## Key Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Full Development Workflow
```bash
# 1. Data Collection (modify word_name variable in script before running)
python capture_samples.py

# 2. Keypoint Processing (configure word_ids list in script before running)  
python create_keypoints.py

# 3. Model Training (uses K-fold cross-validation)
python training_model.py

# 4. Real-time Evaluation
python evaluate_model.py

# 5. Web Server for Video Upload API
python server.py
```

### Development Dependencies
- TensorFlow 2.10.1 with GPU support
- MediaPipe 0.10.11 for pose estimation
- OpenCV 4.9.0.80 for computer vision
- Flask 3.0.2 for web API
- Keras Tuner for hyperparameter optimization (optional)

## Architecture Overview

### Core Pipeline Components

1. **Data Capture Pipeline** (`capture_samples.py`):
   - Webcam-based gesture recording with MediaPipe hand detection
   - Automatic hand presence validation for quality control
   - Frame sequence storage in `frame_actions/{gesture_name}/sample_{timestamp}/`
   - Configurable timeout handling for interrupted gestures

2. **Feature Extraction** (`create_keypoints.py`):
   - MediaPipe Holistic processing of captured frames
   - Extraction of 1,662 keypoints per frame:
     - Pose: 33 landmarks × 4 coords (x,y,z,visibility) = 132 points
     - Face: 468 landmarks × 3 coords (x,y,z) = 1,404 points  
     - Left Hand: 21 landmarks × 3 coords = 63 points
     - Right Hand: 21 landmarks × 3 coords = 63 points
   - HDF5 storage format for efficient data access
   - Zero-padding for missing landmark detection

3. **Neural Network Architecture** (`model.py`):
   ```python
   # Current active model structure:
   Bidirectional LSTM(64) + BatchNorm + Dropout(0.4)
   LSTM(64) + BatchNorm + Dropout(0.4)
   Dense(64, ReLU) + BatchNorm + Dropout(0.4)
   Dense(64, ReLU)
   Dense(n_classes, Softmax)
   ```
   - L2 regularization (0.001) throughout network
   - Adam optimizer with adaptive learning rate
   - Supports multiple sequence lengths (7, 12, 18 frames)

4. **Training System** (`training_model.py`):
   - K-fold cross-validation (5 folds) for robust model evaluation
   - Comprehensive metric generation: Accuracy, Precision, Recall, F1, MCC, Cohen's Kappa
   - Automated visualization generation (confusion matrices, ROC curves, training plots)
   - GPU memory management and early stopping callbacks
   - ReduceLROnPlateau for learning rate optimization

5. **Real-time Recognition** (`evaluate_model.py`):
   - Adaptive model selection based on gesture sequence length:
     - 5-7 frames → Model 7
     - 8-12 frames → Model 12  
     - 13+ frames → Model 18
   - Confidence threshold filtering (configurable, default 0.6)
   - Text-to-speech feedback via gTTS
   - Frame capture during recognition for debugging

6. **Web API Service** (`server.py`):
   - Flask-based REST API for video upload processing
   - Video processing pipeline integration
   - Response formatting for external applications

### Gesture Classes (18 total)
```
acceso, barra de herramientas, borde exterior, cancelar, click derecho,
comando, computacion, contrasenia, copiar texto, correo electronico,
cortar, dibujar tabla, disco duro, escape, pantalla, regresar,
seleccionar, software
```

### Data Organization Structure
```
frame_actions/{gesture_name}/sample_{timestamp}/
├── 1.jpg, 2.jpg, ..., N.jpg    # Raw captured frames
data/keypoints/{gesture_name}.h5  # Processed keypoint sequences  
models/actions_{7|12|18}.keras    # Trained models by sequence length
graphics/                         # Training visualizations and metrics
├── confusion_matrix_fold_{1-5}.png
├── roc_curve_matrix_fold_{1-5}_model_18.png
├── training_plots_fold_{1-5}.png
└── class_specificity_fold_{1-5}.png
```

### Key Configuration (`constants.py`)
- `LENGTH_KEYPOINTS = 1662` - Total keypoints per frame
- `MIN_LENGTH_FRAMES = 5` - Minimum gesture sequence length
- `MODEL_NUMS = [18]` - Active model configurations (currently 18-frame model)
- Path constants for all major directories

## Development Workflow

### Adding New Gestures
1. Modify `word_name` variable in `capture_samples.py`
2. Capture multiple samples (recommended: 15-20 per gesture)
3. Add gesture name to `word_ids` list in `create_keypoints.py`
4. Run keypoint extraction and training pipeline

### Model Training Configuration
- Training uses 5-fold cross-validation for robust evaluation
- Models are saved with sequence length suffix (actions_18.keras)
- Comprehensive metrics saved to `graphics/` folder
- GPU memory growth configured automatically

### Helper Functions (`helpers.py`)
- `mediapipe_detection()` - RGB conversion and MediaPipe processing
- `extract_keypoints()` - Landmark coordinate extraction with zero-padding
- `get_sequences_and_labels()` - Data loading for training
- `pad_sequences()` - Sequence length normalization for inference

### Debugging and Monitoring
- GPU availability and memory configuration handled automatically
- Comprehensive logging during training phases
- Real-time frame capture during evaluation stored in `captures_real_time/`
- Detailed performance metrics visualization

### Technical Design Patterns
**Architecture Patterns:**
- **Separation of Concerns:** Specialized modules for data capture, feature extraction, training, and evaluation
- **Factory Pattern:** Dynamic model creation based on sequence length parameters  
- **Strategy Pattern:** Adaptive model selection based on gesture duration
- **Pipeline Pattern:** Structured data flow from raw frames to predictions

**Performance Optimizations:**
- **GPU Acceleration:** Automatic GPU memory growth configuration
- **Data Efficiency:** HDF5 format for optimized keypoint storage and retrieval
- **Regularization Stack:** L2 + BatchNormalization + Dropout (0.4) for robust training
- **Adaptive Learning:** ReduceLROnPlateau callbacks for training optimization
- **Memory Management:** Efficient sequence padding and batch processing
- **Multi-Model Architecture:** Separate models for different sequence lengths to reduce computational overhead

### Code Quality Standards
- Consistent Spanish documentation throughout codebase
- Modular, reusable function design
- Centralized configuration management via `constants.py`
- Comprehensive error handling for critical operations
- Extensive visualization and metrics generation for model evaluation

### Research & Academic Context
This implementation demonstrates professional-level deep learning practices with:
- Rigorous K-fold cross-validation methodology
- Comprehensive metric suite (MCC, Cohen's Kappa, class specificity)
- Academic-quality visualization generation
- Integration with hyperparameter optimization frameworks (Keras Tuner)
- Potential for real-world deployment and mobile applications