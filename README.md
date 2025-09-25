# Nuros Cookbook 🚀

A comprehensive guide and collection of examples for getting started with GPU computing on the Nuros server using Slurm job scheduling.

## What is Nuros?

Nuros is our high-performance computing server equipped with:
- **2x NVIDIA L40 GPUs** - Perfect for deep learning and AI workloads
- **Slurm job scheduler** - Efficiently manages and queues GPU jobs
- **Python ecosystem** - Pre-configured for machine learning workflows

## Quick Start

**New to GPU computing or Slurm?** Start here:

1. 📖 Read the [Typical Workflow Guide](docs/typical-workflow.md) 
2. 🎯 Try the [MNIST Example](examples/mnist_example.ipynb) 
3. 🔧 Use a [Slurm Template](templates/) to submit your first job
4. 📚 Explore the [Documentation](docs/) for deeper understanding

## Repository Structure

```
nuros_cookbook/
├── examples/           # Complete working examples
│   ├── mnist_example.ipynb        # Simple feed-forward network (beginner-friendly)
│   └── food101_example.py         # Advanced: EfficientNet on Food101
├── templates/                     # Ready-to-use templates
│   └── basic.sbatch               # Minimal GPU job template
├── docs/                          # Documentation and guides
│   ├── slurm-basics.md            # Understanding Slurm commands
│   ├── typical-workflow.md        # Recommended development process
│   ├── notebook-to-script.md      # Converting notebooks to scripts
│   ├── data-management.md         # Getting data onto Nuros
│   ├── examples-walkthrough.md    # Detailed tutorials for all examples
│   ├── troubleshooting.md         # TODO: Common issues and solutions
│   └── resources.md               # Additional learning resources
└── README.md                      # This file
```

## Examples Overview

### 🎯 Beginner: MNIST Digit Classification
- **What**: Handwritten digit recognition with a simple neural network
- **GPU Usage**: Minimal (perfect for learning)
- **Time**: ~10 minutes
- **Files**: `examples/mnist_example.ipynb`, `examples/mnist_example.py`

### 🔥 Advanced: Food101 Classification  
- **What**: Food image classification using EfficientNet
- **GPU Usage**: Heavy (utilizes full GPU capabilities)
- **Time**: ~1-2 hours
- **Files**: `examples/food101_example.py`

## Slurm Templates

### Basic Job Template (`templates/basic.sbatch`)
- Minimal resource request
- Single GPU
- Short time limit
- Perfect for testing and small jobs

### TODO: Heavy Compute Template (`templates/heavy_compute.sbatch`)  
- Maximum resource allocation
- Multiple GPUs if needed
- Extended time limits
- For more expensive training runs

## Documentation Highlights

- **[Typical Workflow](docs/typical-workflow.md)** - Step-by-step process from idea to results
- **[Slurm Basics](docs/slurm-basics.md)** - Essential commands and concepts
- **[Notebook to Script](docs/notebook-to-script.md)** - Converting development code for production
- **[Data Management](docs/data-management.md)** - Getting your data onto the server

## Getting Help

- **Quick Questions**: Check [Troubleshooting](docs/troubleshooting.md)
- **Slurm Commands**: See [Slurm Basics](docs/slurm-basics.md)  
- **Examples Not Working**: Review [Template Usage](templates/README.md)

## Contributing

Found an issue or have a great example to add? 
- Open an issue for problems or questions
- Submit a pull request for improvements
- Share your successful workflows with others

## Quick Command Reference

```bash
# Submit a job
sbatch templates/basic.sbatch

# Check job status  
squeue -u $USER

# View job details (NOT CURRENTLY SUPPORTED)
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,MaxRSS,GPUUtilization

# Cancel a job
scancel <job_id>

# Check GPU availability
sinfo -o "%20N %10c %10m %25f %10G"
```

---

**Ready to get started?** Jump to [Typical Workflow Guide](docs/typical-workflow.md) or try the [MNIST Example](examples/mnist_example.ipynb)!
