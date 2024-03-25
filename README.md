# CNN-Based Image Recognition on NaturalImageNet

## Project Overview
This project focuses on developing and optimizing a deep learning model using Convolutional Neural Networks (CNNs) for image classification. Targeting the NaturalImageNet dataset, a subset of ImageNet containing 20 classes of animals, this project aims to push the boundaries of image recognition in challenging and diverse real-world scenarios. Detailed explanations and written answers regarding the methodology and results are available in the markdown cells of the accompanying Jupyter Notebook (ipynb)

## Requirements
- Python 3.10
- Dependencies listed in `requirements.txt`.

## Installation
1. Clone the repository:
git clone [repository URL]
2. Navigate to the project directory:
cd [Project Directory Name]
3. Install required dependencies:
pip install -r requirements.txt


## Usage
- Run Python scripts and Jupyter notebooks to preprocess the dataset, train, and evaluate the model.
- Utilize `torchvision` for data loading and augmentations, and `torch.nn` for CNN architectures.

## Data and Preprocessing
- The dataset comprises 20 animal classes from NaturalImageNet, a subset of ImageNet.
- Preprocessing includes resizing, random flipping, rotation, and color jitter for augmentations to enhance diversity and robustness.

## Model Architecture and Engineering Decisions
- A custom ResNet-34 architecture is implemented to capture a wide variety of features.
- Augmentation of training data and increased model depth are key engineering decisions.
- Lower batch size and optimized Adamax optimizer are used for better regularization and efficient learning.
- The learning rate and weight decay are finely tuned for balance and overfitting prevention.

## Training and Optimization Strategy
- Hyperparameter tuning combines research, manual exploration, and grid search.
- Manual searches investigate major parameters (architecture depth), while automated grid searches fine-tune minor parameters (learning rate, weight decay).

## Model Performance and Analysis
- The model achieves significantly improved performance compared to baseline, with an F1-score and accuracy optimized for NaturalImageNet.
- Visualization tools provide insights into model decisions, like confusion matrices and incorrectly classified instance visualizations.

## Out of Distribution Evaluation
- Reflective exercise on the model's performance on cartoon versions of the 20 animal classes reveals limitations in handling out-of-distribution data.
- Proposed method for improvement includes incorporating diverse data representation during training for enhanced generalizability.

## Contributing
Contributions to the project are welcome! Please refer to the contributing guidelines for more information on participating.

## Project Status
The project represents an advanced exploration in CNN-based image recognition, with a focus on optimizing model performance for NaturalImageNet. It is currently in a complete state, but open to future expansions and improvements.

---
