# Typical Workflow

The standard process for developing and running machine learning experiments on Nuros.

## Overview

This workflow takes you from initial experimentation to production training runs on GPU resources. The goal is reproducible, efficient research that scales from prototypes to full experiments.

## The 4-Step Process

### 1. Prototype Quickly
**Goal**: Experiment and prototype interactively

- Use Jupyter notebooks for exploration and development
- Work with small datasets, subsets of larger ones, or sample data
- Test ideas quickly with short training runs
- Iterate on model architecture and hyperparameters

**You're ready for step 2 when**: Your approach works and you want to scale up.

### 2. Convert to Script  
**Goal**: Make your code non-interactive and reproducible. [Converting Notebooks to scritps](../docs/notebook-to-script.md)

```bash
# Automatic conversion
jupyter nbconvert --to python your_notebook.ipynb

# Manual conversion for more control
# Replace plt.show() with plt.savefig()
# Add proper file paths and argument parsing
```

### 3. Prepare for Slurm
**Goal**: Configure resources and job submission

- Choose appropriate template from `templates/`
    - or build your own!
- Estimate time and GPU requirements
- Set up organized output directories

```bash
# Copy and modify a template
cp templates/basic_slurm.sbatch my_experiment.sbatch

# Edit resource requirements
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
```

**You're ready for step 4 when**: Your `.sbatch` file is configured.

### 4. Submit & Monitor
**Goal**: Execute training and collect results

```bash
# Submit the job
sbatch my_experiment.sbatch

# Monitor progress
squeue -u $USER
tail -f logs/my_experiment_12345.out

# Collect results when complete
ls ~/runs/my_experiment/
```

**You're done when**: Training completes and results are saved to your organized output directory.

## File Organization Pattern

```
~/your_project/
├── notebooks/              # Step 1: Development
│   └── exploration.ipynb
├── scripts/                # Step 2: Production code  
│   └── train_model.py
├── jobs/                   # Step 3: Slurm scripts
│   └── my_experiment.sbatch
├── logs/                   # Step 4: Job outputs
│   └── my_experiment_12345.out
└── ~/runs/my_experiment/   # Step 4: Results
    ├── checkpoints/
    └── final_model.keras
```

## Quick Decision Points

**Interactive vs Batch?**
- Interactive (`srun`): Testing, debugging, < 30 minutes
- Batch (`sbatch`): Production training, > 30 minutes

**Resource Estimation?**  
- Start conservative, monitor usage, adjust for next run
- Requesting less resources can make your job execute with higher priority
- Time: Add 25% buffer to your estimate

---

**Next Steps**: Try the [MNIST Example](../examples/mnist_example.ipynb) or check out our [Slurm Templates](../templates/).