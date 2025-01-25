# EMNIST Balanced Classification with CNN

## Project Description

This project implements a convolutional neural network (CNN) for image classification using the EMNIST Balanced dataset. This dataset is an extended version of MNIST and contains 47 classes of characters, including letters and digits.

The model is trained using PyTorch and includes:

- Data augmentation
- Convolutional layers with BatchNorm
- Fully connected layers with BatchNorm
- SGD optimizer with momentum
- CrossEntropyLoss with Label Smoothing
- Validation after each epoch

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <repository URL>
cd <directory name>
```

### 2. Install Dependencies

Creating a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # for Linux/macOS
venv\Scripts\activate  # for Windows
```

Installing dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run Training

If you have Docker installed, you can run a container with the environment:

```bash
docker build -t emnist_cnn .
docker run --gpus all -it --rm emnist_cnn
```

Or manually start training:

```bash
python main.py
```

### 4. Evaluate Model Accuracy

After training, you can test the model's accuracy on the test dataset:

```bash
python main.py --test
```

## Dataset

We use **EMNIST Balanced**, which consists of 47 classes (letters and digits). The dataset is automatically loaded via `torchvision.datasets.EMNIST`.

## Model Architecture

- **3 convolutional layers** with BatchNorm and LeakyReLU
- **MaxPooling after each convolutional layer**
- **3 fully connected layers** with BatchNorm and LeakyReLU
- **Final output layer** with 47 neurons

## Results

The model was trained for 10 epochs and achieved **90.27% accuracy** on the test data, which is an excellent result for this task.

## Authors

Aleksey Zhuravlev a.o.zhuravlev@gmail.com