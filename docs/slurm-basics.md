# Slurm Basics

A beginner-friendly guide to understanding and using Slurm (Simple Linux Utility for Resource Management) on the Nuros server.

## What is Slurm?

Slurm is a **job scheduler** that manages who gets to use the GPUs and when. Think of it like a queue system at a busy restaurant:

- You submit a "reservation" (job) with your requirements
- Slurm finds an available "table" (GPU) that meets your needs  
- Your job runs when resources become available
- You get notified when it's done

## Why Do We Need Slurm?

**The Problem**: Multiple people want to use the same GPUs at the same time.

**The Solution**: Slurm ensures fair access by:
- Queuing jobs when resources are busy
- Preventing conflicts between users
- Tracking resource usage
- Automatically starting jobs when resources become available

## Key Concepts

### Jobs
A **job** is a request to run your code with specific resources:
```bash
# This creates a job that requests 1 GPU for 1 hour
sbatch --gres=gpu:1 --time=01:00:00 my_script.py
```

### Nodes
A **node** is a physical computer. Nuros has nodes with GPUs that you can request.

### Job States
- **PENDING (PD)**: Waiting for resources
- **RUNNING (R)**: Currently executing  
- **COMPLETED (CD)**: Finished successfully
- **FAILED (F)**: Finished with errors
- **CANCELLED (CA)**: Stopped by user or admin

## Essential Commands

### Submitting Jobs

#### `sbatch` - Submit a Batch Job
```bash
# Basic submission
sbatch my_job.sbatch

# Submit with inline parameters
sbatch --gres=gpu:1 --time=30:00 --wrap="python train.py"

# Submit and capture job ID
job_id=$(sbatch --parsable my_job.sbatch)
echo "Submitted job: $job_id"
```

#### `srun` - Run Interactive Job
```bash
# Get an interactive session with 1 GPU for 30 minutes
srun --gres=gpu:1 --time=30:00 --pty bash

# Run a single command with GPU access
srun --gres=gpu:1 python quick_test.py
```

### Monitoring Jobs

#### `squeue` - Check Job Status
```bash
# See all your jobs
squeue -u $USER

# See all jobs in the system
squeue

# Watch your jobs update in real-time
watch squeue -u $USER

# See detailed info about a specific job
squeue -j 12345 -o "%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R"
```

#### `sacct` - Job History and Details
```bash
# See your recent jobs
sacct -u $USER

# Detailed info about a specific job
sacct -j 12345 --format=JobID,JobName,State,ExitCode,MaxRSS,GPUUtilization

# See jobs from the last 24 hours
sacct -S $(date -d '1 day ago' +%Y-%m-%d)
```

### Managing Jobs

#### `scancel` - Cancel Jobs
```bash
# Cancel a specific job
scancel 12345

# Cancel all your jobs
scancel -u $USER

# Cancel jobs by name
scancel --name=my_experiment
```

### System Information

#### `sinfo` - Node and Partition Info
```bash
# See available nodes and their status
sinfo

# See GPU information
sinfo -o "%20N %10c %10m %25f %10G"

# Check specific partition
sinfo -p gpu
```

## Job Script Basics (`.sbatch` files)

Every Slurm job needs a script that tells it what to do. Here's the basic structure:

```bash
#!/bin/bash

#SBATCH --job-name=my_experiment      # Job name (appears in squeue)
#SBATCH --output=results_%j.out       # Output file (%j = job ID)
#SBATCH --error=results_%j.err        # Error file
#SBATCH --time=02:00:00              # Time limit (2 hours)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mem=16G                    # Request 16GB RAM
#SBATCH --cpus-per-task=4            # Request 4 CPU cores

# Your commands go here
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

# Load any modules or activate environments
# module load python/3.9
# source activate myenv

# Run your actual code
python train_model.py

echo "Job completed at: $(date)"
```

## Resource Specifications

### GPU Resources
```bash
#SBATCH --gres=gpu:1              # Request 1 GPU (any type)
#SBATCH --gres=gpu:L40:1          # Request 1 L40 GPU specifically  
#SBATCH --gres=gpu:2              # Request 2 GPUs
```

### Memory and CPU
```bash
#SBATCH --mem=32G                 # Total memory for job
#SBATCH --mem-per-cpu=4G          # Memory per CPU core
#SBATCH --cpus-per-task=8         # Number of CPU cores
```

### Time Limits
```bash
#SBATCH --time=30:00              # 30 minutes
#SBATCH --time=02:00:00           # 2 hours
#SBATCH --time=1-12:00:00         # 1 day, 12 hours
```

## Common Scenarios
_Remember that requesting more resources will hurt job queue priority!_

### Quick Testing (< 30 minutes)
```bash
#SBATCH --time=30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
```

### Medium Training (1-4 hours)
```bash
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
```

### Heavy Training (8+ hours)
```bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
```

## Understanding Output

### `squeue` Output Explained
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
12345       gpu my_train    jettd  R      15:30      1 node01
12346       gpu data_prep   jettd PD       0:00      1 (Resources)
```

- **JOBID**: Unique job identifier
- **ST**: State (R=Running, PD=Pending, etc.)
- **TIME**: How long it's been running
- **REASON**: Why pending jobs aren't running yet

### Common Pending Reasons
- **(Resources)**: Not enough GPUs/memory available
- **(Priority)**: Other jobs have higher priority
- **(QOSMaxJobsPerUserLimit)**: You've hit your job limit

## Environment Variables in Jobs

Slurm provides useful environment variables:

```bash
echo "Job ID: $SLURM_JOB_ID"
echo "Node name: $SLURMD_NODENAME"  
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
```

## Best Practices

### Resource Estimation
1. **Start conservative** - Request what you think you need
2. **Monitor usage** - Check if you're using what you requested
3. **Adjust accordingly** - Request more/less for future jobs

### Job Organization
```bash
# Use meaningful job names
#SBATCH --job-name=mnist_lr0.01_batch32

# Organize output files
#SBATCH --output=logs/mnist_%j.out
#SBATCH --error=logs/mnist_%j.err
```

### Debugging Jobs
1. **Test locally first** - Don't debug on the cluster
2. **Use short time limits** for testing
3. **Check error files** if jobs fail
4. **Start with minimal resources** and scale up

## Common Mistakes to Avoid

1. **Requesting too much time** - Hurts scheduling priority
2. **Not saving results** - Jobs can be killed, save frequently  
3. **Ignoring error files** - They contain useful debugging info
4. **Running interactive jobs too long** - Use batch jobs for long tasks
5. **Not cleaning up** - Remove old output files periodically

## Quick Command Cheatsheet

```bash
# Submit job
sbatch job.sbatch

# Check my jobs  
squeue -u $USER

# Cancel job
scancel 12345

# Job details
sacct -j 12345

# System status
sinfo

# Interactive GPU session
srun --gres=gpu:1 --time=30:00 --pty bash
```

## Getting Help

- **Job failing?** Check the `.err` file first
- **Long queue times?** Check `sinfo` for available resources
- **Resource questions?** Look at similar successful jobs with `sacct`
- **Still stuck?** Ask for help with specific error messages

## Next Steps

- **Ready to submit jobs?** Check out our [templates](../templates/)
- **Need to convert a notebook?** See [notebook-to-script.md](notebook-to-script.md)
- **Want the full workflow?** Read [typical-workflow.md](typical-workflow.md)
