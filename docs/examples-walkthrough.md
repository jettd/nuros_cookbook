# Examples Walkthrough

Follow along tutorial instructions for the [examples](../examples/).

# Example List:

Notebook examples:
- N1: [mnist_example](../examples/mnist_example.ipynb)

Script examples:
- S1: [food101_example](../examples/food101_example.py)

### N1: MNIST Example Walkthrough

**What is MNIST?**
MNIST is a classic dataset of 70,000 handwritten digit images (0-9), each 28x28 pixels in grayscale. It's the "Hello World" of computer vision - simple enough to understand quickly, but complex enough to demonstrate real machine learning concepts. One example use case could be automating mail processing which requires classifying letters and digits.

**What you'll learn:**
- Loading and exploring image data
- Preprocessing data for neural networks
- Building a feedforward neural network with Keras
- Training a model and monitoring its progress
- Evaluating model performance
- Saving and loading trained models
- Making predictions on new data

**Expected results:** You should achieve around 90% accuracy on the test set, which means the model correctly identifies ~90 out of 100 handwritten digits.

---

### Steps to Complete:

#### Cell 1: Import Basic Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
```

**What's happening:** Loading essential Python libraries for data manipulation (numpy, pandas), visualization (matplotlib), and utilities. Setting a random seed ensures reproducible results - you'll get the same "random" numbers each time you run the notebook.

---

#### Cell 2: Import Deep Learning Libraries
```python
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
```

**What's happening:** Loading TensorFlow/Keras components for building neural networks. Keras is a high-level API that makes building neural networks much simpler than writing raw TensorFlow code.

**Key concepts:**
- **Sequential**: A linear stack of layers (input → hidden → output)
- **Dense**: Fully connected layers where every neuron connects to every neuron in the next layer
- **Dropout**: A regularization technique that prevents overfitting

---

#### Cell 3: Load the MNIST Dataset
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

**What's happening:** Keras automatically downloads and loads the MNIST dataset, splitting it into training (60,000 images) and test (10,000 images) sets.

**Key concepts:**
- **X_train/X_test**: The actual image data (features)
- **y_train/y_test**: The labels (what digit each image represents)
- **Train/Test split**: We train the model on one set and evaluate on another to test real-world performance

**Expected output:** 
- Training images: (60000, 28, 28)
- Test images: (10000, 28, 28)
- Pixel values range from 0 (black) to 255 (white)

---

#### Cell 4: Visualize Sample Images
```python
plt.figure(figsize=(10, 6))
for i in range(9):
    plt.subplot(3, 3, i+1)
    idx = i * 1000
    plt.imshow(X_train[idx], cmap='gray', interpolation='none')
```

**What's happening:** Displaying 9 sample handwritten digits to see what the data looks like. Uses matplotlib to create a 3x3 grid of images. Building AI/DL models requires intimate knowledge of the data--ususally the first step is to simply see it!

**Expected output:** A 3x3 grid showing various handwritten digits with their labels. Images should look like recognizable numbers but with varying handwriting styles.

---

#### Cell 5: Preprocess the Data - Reshaping
```python
X_train_flat = X_train.reshape(60000, 784)  # 28*28 = 784
X_test_flat = X_test.reshape(10000, 784)
```

**What's happening:** Neural networks expect 1D vectors, not 2D images. We're "flattening" each 28x28 image into a single 784-element vector (like unrolling a grid into a long line).

**Key concepts:**
- **Flattening**: Converting 2D images to 1D vectors
- **784 features**: Each pixel becomes a separate input feature

**Expected output:** Shape changes from (60000, 28, 28) to (60000, 784).

---

#### Cell 6: Preprocess the Data - Normalization
```python
X_train_norm = X_train_flat.astype('float32') / 255.0
X_test_norm = X_test_flat.astype('float32') / 255.0
```

**What's happening:** Converting pixel values from 0-255 range to 0-1 range. Neural networks work much better with small, normalized numbers.

**Key concepts:**
- **Normalization**: Scaling data to a standard range
- **Why normalize**: Helps the model learn faster and more stably

**Expected output:** Pixel values now range from 0.0 to 1.0 instead of 0 to 255.

---

#### Cell 7: Convert Labels to One-Hot Encoding
```python
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)
```

**What's happening:** Converting labels from single numbers (like `3`) to arrays like `[0,0,0,1,0,0,0,0,0,0]` where the position indicates the digit. This format works better for multi-class classification.

**Key concepts:**
- **One-hot encoding**: Representing categories as binary vectors
- **Why one-hot**: Allows the model to output probabilities for each possible digit

**Expected output:** Labels change from shape (60000,) to (60000, 10).

---

#### Cell 8: Build the Neural Network
```python
model = Sequential([
    Input(shape=(784,)),           # Input layer
    Dense(512, activation='relu'), # Hidden layer 1
    Dropout(0.2),                  # Regularization
    Dense(512, activation='relu'), # Hidden layer 2
    Dropout(0.2),                  # More regularization
    Dense(10, activation='softmax') # Output layer
])
```

**What's happening:** Creating a feedforward neural network with 2 hidden layers.

**Key concepts:**
- **Input layer**: Accepts 784 features (one per pixel)
- **Hidden layers**: 512 neurons each with ReLU activation (introduces non-linearity)
- **Dropout**: Randomly ignores 20% of neurons during training to prevent overfitting
- **Output layer**: 10 neurons (one per digit) with softmax activation (outputs probabilities)

**Expected output:** Model summary showing layers, parameters, and total trainable parameters (~670K).

---

#### Cell 9: Compile the Model
```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

**What's happening:** Configuring how the model will learn.

**Key concepts:**
- **Loss function**: Calculates how wrong the model's predictions are
- **Optimizer**: Algorithm that adjusts weights to minimize loss (Adam is adaptive and popular)
- **Metrics**: What to track during training (accuracy = % correct predictions)

**Expected output:** Confirmation that model is compiled and ready to train.

---

#### Cell 10: Train the Model
```python
history = model.fit(
    X_train_norm, Y_train,
    batch_size=128,
    epochs=10,
    verbose=1,
    validation_split=0.1
)
```

**What's happening:** Actually training the neural network on the data.

**Key concepts:**
- **Batch size**: Process 128 examples at once (balances speed and memory usage)
- **Epochs**: Go through the entire dataset 10 times
- **Validation split**: Use 10% of training data to monitor performance during training

**Expected output:** Progress bars showing loss and accuracy improving over 10 epochs. Training accuracy should reach ~98%, validation accuracy ~97%.

**What to watch for:** 
- Loss should generally decrease
- Accuracy should generally increase
- Validation metrics should track training metrics (if they diverge significantly, you might be overfitting)

---

#### Cell 11: Evaluate on Test Set
```python
test_loss, test_accuracy = model.evaluate(X_test_norm, Y_test, verbose=0)
```

**What's happening:** Testing the trained model on completely unseen data to get an honest assessment of performance.

**Key concepts:**
- **Test set**: Data the model has never seen during training
- **Generalization**: How well the model performs on new data

**Expected output:** Test accuracy around 90%. This should be similar to (but often slightly lower than) validation accuracy.

---

#### Cell 12: Visualize Training Progress
```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```

**What's happening:** Creating plots to visualize how the model learned over time.

**Key concepts:**
- **Training curves**: Show model performance over epochs
- **Overfitting signs**: Training accuracy much higher than validation accuracy

**Expected output:** Two plots showing accuracy and loss curves. Both should show improvement over epochs, with validation metrics tracking training metrics reasonably closely.

---

#### Cell 13: Make Predictions and Visualize Results
```python
predictions = model.predict(X_test_norm[:10])
predicted_digit = np.argmax(predictions[i])
```

**What's happening:** Using the trained model to classify new images and visualizing the results with confidence scores.

**Key concepts:**
- **Predictions**: Model outputs probabilities for each digit (0-9)
- **argmax**: Finds the digit with highest probability
- **Confidence**: How sure the model is about its prediction

**Expected output:** Grid showing 10 test images with predicted vs. actual labels. Most should be correct (green), with confidence scores typically above 0.9 for correct predictions.

---

#### Cell 14: Save the Trained Model
```python
model.save('mnist_model.h5')
model.save('mnist_model.keras')
```

**What's happening:** Saving the trained model to disk so you can use it later without retraining.

**Key concepts:**
- **Model persistence**: Saving trained weights and architecture
- **File formats**: .keras (newer) and .h5 (older but widely compatible)

**Expected output:** Confirmation messages and model files created in your directory.

---

#### Cell 15: Load and Test Saved Model
```python
loaded_model = load_model('mnist_model.keras')
```

**What's happening:** Demonstrating how to load a previously saved model and verify it works identically.

**Key concepts:**
- **Model loading**: Restoring a complete trained model
- **Verification**: Ensuring loaded model gives same predictions

**Expected output:** Confirmation that both original and loaded models give identical predictions.

---

### What You've Accomplished

By completing this notebook, you've:

1. **Loaded and explored** a real-world dataset
2. **Preprocessed data** for machine learning (normalization, reshaping, encoding)
3. **Built a neural network** from scratch using Keras
4. **Trained a model** and monitored its learning progress
5. **Evaluated performance** on unseen test data
6. **Visualized results** and model predictions
7. **Saved and loaded** a trained model for future use

**Final Performance:** Your model should achieve ~90% accuracy, meaning it correctly identifies about 90 out of every 100 handwritten digits - pretty impressive for a relatively simple neural network!

---

### S1: Food101 Example

*[To be completed]*
