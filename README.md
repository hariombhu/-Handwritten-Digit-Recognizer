# Handwritten Digit Recognizer (MNIST) 

A deep learning project that recognizes handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. The model achieves **~98% accuracy** on the test set.  

---

##  Project Overview  
This project implements a CNN using **TensorFlow/Keras** to classify handwritten digits. It includes:  
- Training and validation on MNIST dataset (60,000 training, 10,000 testing images).  
- Data normalization and pipeline optimization (caching + prefetching).  
- Regularization (L2 + Dropout) to prevent overfitting.  
- Model saving/loading in `.keras` format.  
- Single-image prediction with visualization.  

---

##  Model Architecture  
- **Input Layer**: 28×28×3  
- **Conv2D (32 filters, 3×3, ReLU, L2)** → MaxPooling2D  
- **Conv2D (64 filters, 3×3, ReLU, L2)** → MaxPooling2D  
- **Conv2D (128 filters, 3×3, ReLU, L2)**  
- **GlobalAveragePooling2D**  
- **Dropout (0.5)**  
- **Dense (10, Softmax)**  

**Parameters**: ~94k trainable  

---

##  Results  
- **Best Validation Accuracy**: **98.48%** (Epoch 29)  
- **Final Test Accuracy**: **98.29%**  
- Loss reduced to **0.15**  

Training and validation curves were also plotted for accuracy and loss.  

---

##  Project Structure  
```
├── Mnist_dataset/
│   ├── train/ (0–9 subfolders with training images)
│   └── test/  (0–9 subfolders with test images)
├── mnist_cnn.keras          # Saved trained model
├── train.py                 # Training script
├── evaluate.py              # Evaluate on test dataset
├── predict.py               # Single image prediction
├── README.md                # Project documentation
```

---

##  How to Run  

### 1. Clone repo  
```bash
git clone https://github.com/hariombhu/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
```

### 2. Install dependencies  
```bash
pip install tensorflow matplotlib
```

### 3. Train the model  
```bash
python train.py
```

### 4. Evaluate on test dataset  
```bash
python evaluate.py
```

### 5. Predict a single digit  
Update `img_path` in `predict.py` and run:  
```bash
python predict.py
```

---

##  Example Prediction  

Input: handwritten digit `5`  

Prediction: **5** with **99.48% confidence**  

---

##  Technologies Used  
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  

---

##  Future Improvements  
- Deploy model using Flask/FastAPI for real-time prediction  
- Extend to other handwritten datasets (e.g., EMNIST)  
- Try advanced CNN architectures (ResNet, EfficientNet)  
