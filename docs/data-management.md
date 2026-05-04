# Data Management

Getting your data onto Nuros and organizing it for machine learning workflows.

## Getting Data to Nuros

You can transfer files to Nuros using standard secure file transfer methods within your disk quota limits. 

### Specifics: From `Google Drive`

You can easily download public Google Drive content directly to the server using **gdown**:

```bash
# Install once
pip install gdown

# Download a file
gdown https://drive.google.com/uc?id=FILE_ID

# Download an entire folder (recursively)
gdown --folder https://drive.google.com/drive/folders/FOLDER_ID
```

No sign-in is required for links shared as “Anyone with the link can view.”  
The folder or file will appear in your current working directory with the same name as in Google Drive.


### Command Line Options: `Between PC/Laptop and Nuros`

The following (scp/sftp) should be performed in a terminal on the sending device, not while on Nuros.

**scp (Secure Copy)**
```bash
# Copy single file
scp my_dataset.csv username@nuros.unl.edu:/home/username/data/

# Copy entire directory
scp -r my_project_data/ username@nuros.unl.edu:/home/username/data/
```

**sftp (SSH File Transfer Protocol)**
```bash
# Interactive session
sftp username@nuros.unl.edu
sftp> put my_dataset.csv data/
sftp> put -r my_directory/ data/
```

### GUI Applications

**macOS: Cyberduck**
- Free SFTP/SCP client with drag-and-drop interface
- Download: [cyberduck.io](https://cyberduck.io/)
- Connect using SFTP protocol with your Nuros credentials

**Windows: FileZilla**
- Free, cross-platform SFTP client
- Download: [filezilla-project.org](https://filezilla-project.org/)
- Use SFTP protocol (not FTP) with your login details

### Built-in Dataset Downloads

Many datasets download automatically:
```python
# Keras datasets (MNIST, CIFAR, etc.)
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# TensorFlow datasets
import tensorflow_datasets as tfds
dataset = tfds.load('food101', data_dir='~/data/')

# Hugging Face datasets  
from datasets import load_dataset
dataset = load_dataset('imagefolder', data_dir='~/data/my_images')
```

**Large datasets:**
- Download once, reuse across experiments
- Consider compressed formats (`.tar.gz`, `.zip`)
- Clean up old experiment outputs periodically

---

**Next Step**: Once your data is on Nuros, see [Typical Workflow](typical-workflow.md) for development process.