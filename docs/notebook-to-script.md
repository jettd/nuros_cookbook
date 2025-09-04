# Converting Notebooks to Scripts

A quick guide for converting Jupyter notebooks to Python scripts that can run on Nuros.

## Why Convert Notebooks to Scripts?

Notebooks are perfect for exploration and development, but scripts are better for:
- **Slurm job submission** - Batch jobs need standalone scripts
- **Reproducibility** - Scripts run the same way every time
- **Automation** - No manual cell execution required
- **Resource efficiency** - No notebook server overhead

## Automatic Conversion with nbconvert

The fastest way to convert a notebook is using Jupyter's built-in `nbconvert` tool.

### Basic Command
```bash
jupyter nbconvert --to python your_notebook.ipynb
```

This creates `your_notebook.py` with all cells converted to regular Python code.

### Example: Converting the MNIST Notebook
```bash
# Convert the MNIST example from the cookbook
jupyter nbconvert --to python mnist_example.ipynb

# This creates mnist_example.py
ls mnist_example.*
# mnist_example.ipynb  mnist_example.py
```

### Useful nbconvert Options
```bash
# Remove input prompts and cell numbers (cleaner output)
jupyter nbconvert --to python --no-prompt mnist_example.ipynb

# Specify output filename
jupyter nbconvert --to python mnist_example.ipynb --output my_training_script.py

# Remove empty cells and markdown
jupyter nbconvert --to python --no-prompt --RegexRemovePreprocessor.patterns="['^\s*$']" mnist_example.ipynb
```

The converted script will run exactly like your notebook, but non-interactively.

## Manual Conversion Best Practices

Sometimes you need more control than automatic conversion provides. Here's when and how to convert manually:

### When to Go Manual
- **Complex interactive elements** that don't convert well
- **Notebooks with experimental/debug code** you want to clean up
- **Adding command-line arguments** or configuration options
- **Restructuring code** for better organization

### Key Changes to Make

#### 1. Replace Interactive Plots
**Notebook version:**
```python
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.show()  # ← Interactive display
```

**Script version:**
```python
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.savefig('training_accuracy.png', dpi=150, bbox_inches='tight')  # ← Save to file
plt.close()  # Free memory
print("Training plot saved as 'training_accuracy.png'")
```

#### 2. Handle File Paths and Outputs
**Notebook version:**
```python
model.save('my_model.h5')  # Saves in current directory
```

**Script version:**
```python
import os
from pathlib import Path

# Create organized output directory
output_dir = Path("results") / "mnist_experiment"
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / "trained_model.keras"
model.save(model_path)
print(f"Model saved to: {model_path}")
```

#### 3. Add Progress Indicators
**Notebook version:**
```python
# Relies on Jupyter's output display
history = model.fit(X_train, y_train, epochs=10, verbose=1)
```

**Script version:**
```python
import time

print("Starting training...")
start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, verbose=1)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.1f} seconds")
```

#### 4. Replace Display() with Print()
**Notebook version:**
```python
display(model.summary())  # Jupyter-specific
```

**Script version:**
```python
print("Model Architecture:")
model.summary()  # Works everywhere
```

## Common Issues & Solutions

### Problem: "Display" or "IPython" Errors
**Error:** `NameError: name 'display' is not defined`

**Solution:** Replace Jupyter-specific functions:
```python
# Instead of: display(df.head())
print(df.head())

# Instead of: from IPython.display import Image
# Just save images to files instead
```

### Problem: Interactive Widgets Don't Work
**Error:** Widgets like sliders, progress bars show as plain text

**Solution:** Replace with simple print statements:
```python
# Instead of: progress_bar = tqdm(range(100))
# Use: print statements or simple progress indicators

for i, batch in enumerate(data_loader):
    if i % 100 == 0:
        print(f"Processed {i}/{len(data_loader)} batches")
```

### Problem: Relative Paths Break
**Error:** `FileNotFoundError` when script runs from different directory

**Solution:** Use absolute paths or script-relative paths:
```python
import os
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent
data_path = script_dir / "data" / "dataset.csv"

# Or use home directory
home_dir = Path.home()
results_dir = home_dir / "results"
```

## Tips for Success

- **Start simple** - Convert first, optimize later
- **Test locally** - Always run the script on your development machine before submitting to Slurm
- **Save everything** - When in doubt, save plots and data to files
- **Use meaningful names** - `model_epoch_10.keras` is better than `model.keras`
- **Check file paths** - Make sure your script can find its inputs and create its outputs

---

**Ready to submit?** Check out [Slurm Basics](slurm-basics.md) for job submission, or use one of our [templates](../templates/) to get started quickly!
