# Digit Recognition Project

Welcome to the **Digit Recognition Project**! This repository showcases the implementation of a Convolutional Neural Network (CNN) for recognizing handwritten digits from the popular MNIST dataset. With the power of deep learning, this project achieves high accuracy in identifying digits from 0 to 9.

## Project Overview
Handwritten digit recognition is a classic problem in computer vision and machine learning. In this project, we utilize CNNs, a powerful deep learning architecture, to classify images of digits effectively. The MNIST dataset serves as our benchmark dataset, containing 70,000 grayscale images of handwritten digits (28x28 pixels).

## Key Features
- **Efficient Architecture**: Uses layers like Conv2D, MaxPooling2D, Flatten, and Dense to build a robust CNN.
- **Preprocessing**: Includes normalization and reshaping of input images.
- **Accuracy**: Achieves high performance on both training and test data.
- **Customizability**: The architecture can be fine-tuned for better results.

## Dataset
The MNIST dataset contains:
- **Training Data**: 60,000 images
- **Test Data**: 10,000 images
- Each image is 28x28 pixels, grayscale, with labels from 0 to 9.

## CNN Architecture
Our Convolutional Neural Network is designed as follows:
1. **Input Layer**:
   - Accepts 28x28 grayscale images reshaped to (28, 28, 1).
2. **Convolutional Layers (Conv2D)**:
   - Extracts features using learnable filters.
3. **Pooling Layers (MaxPooling2D)**:
   - Reduces spatial dimensions and computational complexity.
4. **Flatten Layer**:
   - Converts the 2D feature maps into a 1D vector for the fully connected layer.
5. **Dense Layers**:
   - Fully connected layers that perform classification.

## Tools and Libraries
This project is built using:
- **Python**: Programming language for implementation.
- **TensorFlow/Keras**: Framework for building and training the CNN.
- **NumPy**: For numerical computations.
- **Matplotlib**: For visualizing results.

## Project Workflow
1. **Data Preprocessing**:
   - Normalize pixel values to the range [0, 1].
   - Reshape images for compatibility with the CNN.
2. **Model Building**:
   - Create a sequential model using Conv2D, MaxPooling2D, Flatten, and Dense layers.
3. **Training**:
   - Train the model on the MNIST training set with a validation split.
4. **Evaluation**:
   - Evaluate model performance on the test dataset.
5. **Prediction**:
   - Test the model with custom inputs or sample images.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Prasanth7733/digit-recognition.git
   ```
2. Navigate to the project folder:
   ```bash
   cd digit-recognition
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python main.py
   ```

## Results
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98%
- **Visualization**: Includes confusion matrices and sample predictions.

## Acknowledgments
- **Dataset**: The MNIST dataset is publicly available and maintained by Yann LeCun and colleagues.
- **Framework**: TensorFlow/Keras simplifies deep learning implementation and experimentation.

---

Feel free to contribute to this project or raise any issues. Let’s explore the exciting world of deep learning together!

