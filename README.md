# Advanced Customizations in Deep Learning and Neural Networks

This repository contains comprehensive Colab notebooks demonstrating advanced deep learning techniques, data augmentation strategies, and custom implementations in both **TensorFlow/Keras** and **PyTorch**.

## Repository Structure

```
.
├── README.md
├── Part1_Regularization_Augmentation/
│   ├── Part1a_Regularization_TensorFlow.ipynb
│   ├── Part1b_Regularization_PyTorch.ipynb
│   ├── Part1c_MonteCarlo_Dropout.ipynb
│   ├── Part1d_Weight_Initializations.ipynb
│   ├── Part1e_Custom_Dropout_Regularization.ipynb
│   ├── Part1f_Callbacks_TensorBoard.ipynb
│   ├── Part1g_Keras_Tuner.ipynb
│   ├── Part1h_KerasCV_Augmentation.ipynb
│   ├── Part1i_Image_Video_Augmentation.ipynb
│   └── Part1j_Text_TimeSeries_Tabular_Speech_Doc.ipynb
│
└── Part2_Advanced_Constructs/
    ├── Part2a_Custom_LR_Scheduler.ipynb
    ├── Part2b_Custom_Loss_Activation_Init_Reg.ipynb
    ├── Part2c_Custom_Metrics.ipynb
    ├── Part2d_Custom_Layers.ipynb
    ├── Part2e_Custom_Models.ipynb
    ├── Part2f_Custom_Optimizer.ipynb
    ├── Part2g_Custom_Training_Loop.ipynb
    └── Part2h_Weights_and_Biases.ipynb
```

---

## Part 1: Data Augmentation and Regularization Techniques

### [Part1a: Regularization Techniques - TensorFlow](Part1_Regularization_Augmentation/Part1a_Regularization_TensorFlow.ipynb)
Demonstrates regularization techniques in TensorFlow/Keras with A/B testing:
- **L1 Regularization** (Lasso) - Promotes sparsity in weights
- **L2 Regularization** (Ridge) - Prevents large weight values
- **Dropout** - Randomly drops neurons during training
- **Early Stopping** - Prevents overfitting by monitoring validation loss
- **Batch Normalization** - Normalizes layer inputs for faster training

### [Part1b: Regularization Techniques - PyTorch](Part1_Regularization_Augmentation/Part1b_Regularization_PyTorch.ipynb)
Same regularization concepts implemented in PyTorch:
- L1/L2 regularization using weight decay and manual implementation
- Dropout layers and training/eval modes
- Early stopping with custom callbacks
- BatchNorm layers and their behavior

### [Part1c: Monte Carlo Dropout](Part1_Regularization_Augmentation/Part1c_MonteCarlo_Dropout.ipynb)
Uncertainty estimation using MC Dropout:
- Understanding epistemic vs aleatoric uncertainty
- Implementing MC Dropout in TensorFlow
- Implementing MC Dropout in PyTorch
- Visualizing prediction uncertainty

### [Part1d: Weight Initializations](Part1_Regularization_Augmentation/Part1d_Weight_Initializations.ipynb)
Comprehensive guide to weight initialization strategies:
- **Xavier/Glorot** - Best for tanh and sigmoid activations
- **He/Kaiming** - Best for ReLU and variants
- **LeCun** - Best for SELU activation
- **Orthogonal** - Good for RNNs
- Comparison and when to use each

### [Part1e: Custom Dropout & Regularization](Part1_Regularization_Augmentation/Part1e_Custom_Dropout_Regularization.ipynb)
Creating custom regularization techniques:
- Custom Dropout layer implementation
- Custom L1/L2 regularizer classes
- Alpha Dropout for SELU networks
- Combining multiple regularization strategies

### [Part1f: Callbacks & TensorBoard](Part1_Regularization_Augmentation/Part1f_Callbacks_TensorBoard.ipynb)
Using callbacks and visualization:
- ModelCheckpoint for saving best models
- EarlyStopping with patience
- LearningRateScheduler
- TensorBoard integration and visualization
- Custom callback creation

### [Part1g: Keras Tuner](Part1_Regularization_Augmentation/Part1g_Keras_Tuner.ipynb)
Hyperparameter optimization with Keras Tuner:
- RandomSearch tuner
- Hyperband tuner
- BayesianOptimization tuner
- Tuning architecture and learning parameters

### [Part1h: KerasCV Data Augmentation](Part1_Regularization_Augmentation/Part1h_KerasCV_Augmentation.ipynb)
Modern image augmentation with KerasCV:
- RandAugment
- CutMix and MixUp
- Random cropping and flipping
- Color jittering and transformations

### [Part1i: Image & Video Augmentation](Part1_Regularization_Augmentation/Part1i_Image_Video_Augmentation.ipynb)
Augmentation using AugLy library:
- Image augmentation techniques
- Video augmentation for temporal data
- Building augmentation pipelines
- Integration with training workflows

### [Part1j: Multi-modal Data Augmentation](Part1_Regularization_Augmentation/Part1j_Text_TimeSeries_Tabular_Speech_Doc.ipynb)
Augmentation for various data types:
- **Text** - Using nlpaug for NLP augmentation
- **Time Series** - Window slicing, jittering, scaling
- **Tabular Data** - SMOTE, noise injection
- **Speech/Audio** - Time stretching, pitch shifting
- **Document Images** - Document-specific transformations

---

## Part 2: Advanced Keras and PyTorch Constructs

### [Part2a: Custom Learning Rate Schedulers](Part2_Advanced_Constructs/Part2a_Custom_LR_Scheduler.ipynb)
Advanced learning rate scheduling:
- OneCycleLR scheduler
- Cosine annealing with warm restarts
- Custom exponential decay
- Learning rate finder implementation

### [Part2b: Custom Loss, Activation, Initializer, Regularizer](Part2_Advanced_Constructs/Part2b_Custom_Loss_Activation_Init_Reg.ipynb)
Building custom components:
- **Custom Loss** - Huber loss, Focal loss
- **Custom Activation** - Leaky ReLU variants, Swish
- **Custom Initializer** - Glorot variants
- **Custom Regularizer** - L1 with custom strength
- **Custom Constraint** - Positive weights constraint

### [Part2c: Custom Metrics](Part2_Advanced_Constructs/Part2c_Custom_Metrics.ipynb)
Creating custom evaluation metrics:
- Huber Metric implementation
- F1 Score metric
- Custom streaming metrics
- Multi-output metrics

### [Part2d: Custom Layers](Part2_Advanced_Constructs/Part2d_Custom_Layers.ipynb)
Building custom neural network layers:
- Simple exponential layer
- MyDense - Custom dense layer
- AddGaussianNoise - Noise injection layer
- LayerNormalization from scratch
- MaxNormDense - Dense with max norm constraint

### [Part2e: Custom Models](Part2_Advanced_Constructs/Part2e_Custom_Models.ipynb)
Creating custom model architectures:
- ResidualBlock implementation
- ResidualRegressor model
- Functional API vs Subclassing
- Multi-input/output models

### [Part2f: Custom Optimizer](Part2_Advanced_Constructs/Part2f_Custom_Optimizer.ipynb)
Implementing custom optimizers:
- MyMomentumOptimizer
- Understanding optimizer internals
- Gradient manipulation techniques
- Comparison with built-in optimizers

### [Part2g: Custom Training Loop](Part2_Advanced_Constructs/Part2g_Custom_Training_Loop.ipynb)
Low-level training control:
- GradientTape in TensorFlow
- Manual forward/backward pass in PyTorch
- Custom training step logic
- Fashion MNIST complete example

### [Part2h: Weights & Biases Integration](Part2_Advanced_Constructs/Part2h_Weights_and_Biases.ipynb)
ML experiment tracking with W&B:
- Setting up W&B projects
- Logging metrics and hyperparameters
- Artifact tracking
- Hyperparameter sweeps

---

## How to Use

1. **Open in Google Colab**: Click on any notebook link above, then click the "Open in Colab" button
2. **Run cells sequentially**: Each notebook is designed to be run from top to bottom
3. **GPU recommended**: For faster training, enable GPU in Colab (Runtime > Change runtime type > GPU)

## Requirements

Most dependencies are pre-installed in Google Colab. Additional installations are handled within each notebook:

```python
# Common installations (done in notebooks)
!pip install keras-cv keras-tuner
!pip install augly nlpaug
!pip install wandb
```

## Video Walkthrough

A detailed video explanation of each Colab notebook is available, covering:
- Line-by-line code explanation
- Execution and output demonstration
- Conceptual understanding of each technique

---

## References

- [Hands-On Machine Learning (3rd Edition)](https://github.com/ageron/handson-ml3)
- [Hands-On Machine Learning PyTorch](https://github.com/ageron/handson-mlp)
- [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)
- [KerasCV Documentation](https://keras.io/keras_cv/)
- [AugLy by Facebook Research](https://github.com/facebookresearch/AugLy)
- [Data Augmentation Review](https://github.com/AgaMiko/data-augmentation-review)

---

## Author

Created as part of the Advanced Deep Learning course assignment.

## License

This project is for educational purposes.
