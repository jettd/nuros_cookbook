# Examples Walkthrough

Follow along tutorial instructions for the [examples](../examples/).

# Example List:

Notebook examples:
- N1: [mnist_example](../examples/mnist_example.ipynb)
    - Simple follow along instructions [here](#n1-mnist-example-walkthrough)

Script examples:
- S1: [food101_example](../examples/food101_example.py)
    - Explaination (of code) and slurm submission instructions in [this section](#s1-food101-example-walkthrough)

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

### S1: Food101 Example Walkthrough

**What is Food101?**
Food101 is a real-world dataset containing 101,000 images across 101 food categories (1,000 images per category). Unlike MNIST's simple 28x28 grayscale digits, these are full-color photographs of actual food dishes with varying lighting, backgrounds, and presentation styles. This represents the jump from toy problems to production-scale computer vision.

**Key Differences from MNIST:**
- **Real dataset**: Download and manage ~5GB of data
- **Production pipeline**: Efficient data loading with tf.data
- **Modern architecture**: EfficientNet instead of simple Dense layers  
- **Hardware optimization**: Mixed precision training, GPU memory management
- **Script format**: Command-line execution instead of interactive notebook
- **Longer training**: 1-2 hours instead of minutes

**What you'll learn:**
- Converting from notebook development to production scripts
- Managing large datasets and file organization
- Using modern CNN architectures (EfficientNet)
- Optimizing training performance with mixed precision
- Submitting and monitoring long-running Slurm jobs
- Proper checkpointing and logging for production training

**Expected results:** ~85% accuracy on the test set after 30 epochs (much harder than MNIST's 10 simple digits).

---
## Understanding the Script Structure

### Hardware & Performance Setup
```python 
mixed_precision.set_global_policy("mixed_float16")  # L40 loves this
strategy = tf.distribute.MirroredStrategy() if num_gpus > 1 else tf.distribute.get_strategy()
```

**What's happening:** Configures the GPU for optimal performance on Nuros. Mixed precision uses both 16-bit and 32-bit floating point numbers - faster training with minimal accuracy loss on modern GPUs.

### Dataset Management
```python
DATA_DIR = HOME / "data" / "food-101"
# Downloads ~5GB dataset once, extracts to organized structure
```

**What's happening:** Unlike MNIST's automatic download, this manages real dataset logistics. Downloads once and reuses, creates organized directory structure in `~/data/food-101/`.

**File Organization Benefits:**
- **Reusable**: Download once, use for multiple experiments
- **Organized**: Clear separation of raw data, extracted files, and results
- **Scalable**: Pattern works for any dataset size

### Modern Data Pipeline
```python
def make_ds(paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), labels))
    if training:
        ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    ds = ds.map(decode_load_resize, num_parallel_calls=4)
    # ... data augmentation, batching, prefetching
```

**What's happening:** Builds an efficient data pipeline that keeps GPUs fed with data. Much more sophisticated than loading everything into memory like MNIST.

**Key concepts:**
- **Parallel loading**: Multiple CPU threads load images while GPU trains
- **Data augmentation**: Random flips and color changes prevent overfitting
- **Prefetching**: Loads next batch while GPU processes current batch

### Model Architecture
```python
base = applications.EfficientNetB4(include_top=False, weights=None, input_tensor=input_shape)
# Note: weights=None means training from scratch (not transfer learning)
```

**What's happening:** Uses EfficientNetB4, a modern CNN architecture designed for efficiency. Much more sophisticated than MNIST's simple Dense layers. Notice that `weights=None` so we are not loading the pretrained model; instead we just load the layers and train from scratch.

---

## Running with Slurm

### Step 1: Prepare Your Environment

**Upload the script** to Nuros (e.g., in your home directory)

### Step 2: Choose the Right Slurm Template

This job needs significant resources and time. Use the heavy compute template as a starting point:

**File: `food101_job.sbatch`**
```bash
#!/bin/bash
#SBATCH --job-name=food101_efficientnet
#SBATCH --time=02:30:00              # 2.5 hours (be generous)
#SBATCH --gres=gpu:1                 # 1 GPU sufficient
#SBATCH --mem=32G                    # Plenty of RAM for data loading
#SBATCH --cpus-per-task=8            # Multiple cores for data pipeline
#SBATCH --output=logs/food101_%j.out
#SBATCH --error=logs/food101_%j.err

# Environment setup (adjust for your Python setup)
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "Starting Food101 training at: $(date)"
echo "Running on node: $SLURMD_NODENAME" 
echo "Job ID: $SLURM_JOB_ID"

# Run the training script
python food101_example.py

echo "Training completed at: $(date)"
```

### Step 3: Submit the Job

```bash
# Submit the job
sbatch food101_job.sbatch

# Check job status
squeue -u $USER

# Monitor progress (job ID from sbatch output)
tail -f logs/food101_12345.out
```

### Step 4: Monitor Progress

**What to watch for in the output:**
- **Dataset download**: First run downloads ~5GB (adds ~10-15 minutes)
- **GPU detection**: Should show "Visible GPUs: 1"
- **Training progress**: Shows loss/accuracy every few epochs (verbose=2)
- **Checkpoints**: Saves model every epoch to `~/runs/food101_efficientnet/B4/`

**Expected output timeline:**
```
# Initial setup (first run only)
Downloading Food-101 (~5GB)...
Extracting Food-101...

# Every run
Visible GPUs: 1
Classes: 101 Train imgs: 74747 Test imgs: 25250
Training samples: 73252, Validation samples: 1495

# Training progress
Epoch 1/30
... loss: 3.2850 - accuracy: 0.2847 - val_loss: 2.8934 - val_accuracy: 0.3845
Epoch 2/30
... loss: 2.7821 - accuracy: 0.4123 - val_loss: 2.5643 - val_accuracy: 0.4521
...
Epoch 30/30
... loss: 0.8234 - accuracy: 0.7823 - val_loss: 1.2345 - val_accuracy: 0.7234

# Final evaluation
Test: [1.1234, 0.8543]  # [loss, accuracy]
Throughput: 45.2 images/sec
```

### Step 5: Check Results

After the job completes, check:

**Checkpoints and logs:**
```bash
ls ~/runs/food101_efficientnet/B4/
# Should contain: ckpt-01.keras, ckpt-02.keras, ..., ckpt-30.keras, log.csv
```

**Training history:**
```bash
head ~/runs/food101_efficientnet/B4/log.csv
# CSV with epoch, loss, accuracy, val_loss, val_accuracy columns
```

**Final test accuracy:** Should be around **85%** in the job output.

---

## Common Issues and Solutions

### Job Fails Immediately
**Check:** Look at the `.err` file for Python import errors or missing dependencies.
**Solution:** Verify your Python environment has TensorFlow/Keras installed.

### "Out of Memory" Errors
**Reduce batch size:** Change `BATCH_PER_GPU = 32` to `BATCH_PER_GPU = 16` in the script.
**More memory:** Request more memory in your sbatch script (`--mem=64G`).
**Export GPU Growth:** In the Nuros terminal, try `export TF_FORCE_GPU_ALLOW_GROTH=true` which changes how the GPUs allocate memory.

### Job Times Out
**Longer time limit:** Increase `--time=03:00:00` in your sbatch script.
**Check progress:** Look at checkpoint files to see how far training got.

### Slow Download on First Run
**Expected:** 5GB download takes 5-15 minutes depending on connection.
**One-time only:** Subsequent runs reuse the downloaded data.

### Poor Performance (< 80% accuracy)
**Check epochs:** Make sure all 30 epochs completed.
**Verify GPU:** Ensure "Visible GPUs: 1" appears in output.
**Normal variation:** Results between 83-87% are typical.

---

## Understanding the Results

### What Good Training Looks Like
Note: these are estimates.
- **Loss decreases** steadily over epochs
- **Accuracy increases** from ~28% (epoch 1) to ~85% (epoch 30)  
- **Validation metrics** track training metrics reasonably closely
- **Test accuracy** similar to final validation accuracy

### Performance Expectations
Note: these are estimates.
- **Throughput**: ~40-60 images/sec on L40 with mixed precision
- **Memory usage**: ~8-12GB GPU memory with batch size 32
- **Final accuracy**: 83-87% on test set (Food101 is challenging!)

### File Organization Results
```
~/data/food-101/          # Dataset (reusable)
├── food-101.tar.gz       # Original download
├── images/               # 101,000 food images
└── meta/                 # train.txt, test.txt splits

~/runs/food101_efficientnet/B4/  # Experiment results
├── ckpt-01.keras         # Model checkpoints
├── ckpt-02.keras         # (one per epoch)
├── ...
├── ckpt-30.keras
└── log.csv              # Training history
```

---

## What You've Accomplished

Moving from MNIST to Food101 represents a major step toward production machine learning:

1. **Real dataset management** - Download, organize, and process large datasets
2. **Modern architecture** - Used state-of-the-art EfficientNet instead of basic layers
3. **Production pipeline** - Efficient data loading that scales to any dataset size
4. **Hardware optimization** - Leveraged mixed precision and GPU memory management
5. **Proper experimentation** - Checkpointing, logging, and organized results
6. **Slurm workflow** - Submitted and monitored long-running training jobs

**You're now ready to tackle real computer vision problems with proper tools and workflows!**
