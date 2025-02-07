# Satellite Image Classification using Custom CNN

## Overview  
This project aims to classify satellite images into four categories: **Cloudy, Desert, Green_Area, and Water** using a **custom Convolutional Neural Network (CNN)**. The dataset consists of 256x256 resolution images, and the model is trained using **TensorFlow/Keras**.  

## Features  
✅ Data preprocessing and augmentation  
✅ Custom CNN model for image classification  
✅ Model evaluation with accuracy, loss plots, and confusion matrix  
✅ Fine-tuning with optimized learning rate and dropout adjustments  
✅ Deployment using Flask for real-time classification  

## Dataset  
The dataset contains satellite images labeled under the following classes:  
- **Cloudy**  
- **Desert**  
- **Green_Area**  
- **Water**  

## Project Structure  
```
├── customcnn.ipynb         # Jupyter Notebook for model training
├── app.py                  # Flask app for model deployment
├── requirements.txt        # Dependencies list
├── model.h5                # Trained CNN model
├── image_dataset.csv       # Processed dataset file
├── static/                 # Directory for images
├── templates/              # HTML templates for web app
├── video.mp4               # Project demonstration video
├── presentation.pptx       # Project presentation
```

## Installation & Usage  

### 1️⃣ Setup Environment  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/yourusername/satellite-image-classification-cnn.git
cd satellite-image-classification-cnn
pip install -r requirements.txt
```

### 2️⃣ Train the Model (Optional)  
To train the model from scratch, run:  
```bash
jupyter notebook
```
Open **customcnn.ipynb** and execute the cells.

### 3️⃣ Run the Web App  
Start the Flask app for real-time predictions:  
```bash
python app.py
```
Access the web app at **http://127.0.0.1:5000/**.

## Model Architecture  
The custom CNN includes:  
- **Convolutional layers** for feature extraction  
- **Batch Normalization & MaxPooling** for stability  
- **Dropout layers** to prevent overfitting  
- **Dense layers** for classification  

## Evaluation Metrics  
- **Test Accuracy:** *Achieved ~XX%*  
- **Confusion Matrix & Classification Report**  
- **Accuracy & Loss Graphs**  

## Results  
📊 High accuracy achieved with optimized training parameters.  
🔍 Fine-tuned the model to mitigate overfitting.  

## Future Enhancements  
🚀 Explore transfer learning with pre-trained models.  
🌍 Extend dataset for better generalization.  

## Acknowledgments  
📌 Dataset: [Kaggle - Satellite Image Classification](https://www.kaggle.com/)  
📌 Frameworks: TensorFlow, Keras, Flask  
