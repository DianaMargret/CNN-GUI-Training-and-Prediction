# README

## **GUI CNN Training and Prediction Software**

This application is a Python-based GUI software for training and predicting using Convolutional Neural Networks (CNNs). It is designed for users who want an intuitive interface to build, train, evaluate, and use image classification models without diving into code. 

---

### **Features**
1. **Graphical User Interface (GUI)**:
   - Built with `tkinter` for ease of use.
   - Allows users to select image folders, set training parameters, and predict images.

2. **Customizable Training**:
   - Supports adjustable epochs, patience, test ratio, and validation ratio.
   - Implements early stopping to prevent overfitting.

3. **Image Classification**:
   - Prepares datasets from user-selected folders containing labeled images.
   - Uses TensorFlow/Keras for model building and training.

4. **Model Management**:
   - Save trained models and their labels as `.h5` files.
   - Load existing models for prediction or further training.

5. **Prediction**:
   - Predicts the class of new images with confidence percentages.

---

### **How to Deploy the Code**

#### **Prerequisites**
1. Install Python 3.8 or higher.
2. Install the required libraries:
   ```bash
   pip install tensorflow pillow numpy scikit-learn
   ```

#### **Steps to Run**
1. Download the `FINAL-BEST-GUI.py` file.
2. Open a terminal or command prompt and navigate to the directory containing the file.
3. Run the script:
   ```bash
   python FINAL-BEST-GUI.py
   ```
4. The GUI will launch, allowing you to interact with the software.

---

### **How to Use**

#### **Step 1: Select Image Folders**
- Click "Add Category Folder" to select folders containing labeled images (e.g., `cats`, `dogs`).
- Ensure each folder contains images in `.jpg`, `.jpeg`, or `.png` format.

#### **Step 2: Set Training Parameters**
- Specify:
  - Epochs: Number of iterations for training.
  - Patience: Early stopping patience.
  - Test Ratio: Percentage of data used for testing.
  - Validation Ratio: Percentage of data used for validation.

#### **Step 3: Train Model**
- Click "Train Model" to start training. The GUI will display progress and evaluation results (accuracy and loss).

#### **Step 4: Save or Load Model**
- Save the trained model using "Save Model."
- Load a previously saved model using "Load Model."

#### **Step 5: Predict New Images**
- Click "Predict Image" and select an image file. The software will display the predicted class along with confidence percentages.

---

### **Specialty of This Code**

This software simplifies CNN-based image classification by providing a user-friendly GUI. It is ideal for beginners in machine learning who want to experiment with deep learning without writing complex code. The application is powered by TensorFlow/Keras and includes features like dataset preparation, model training, evaluation, saving/loading models, and real-time prediction.

---

### **Convert to Executable (.exe) Using PyInstaller**

To distribute this software as an executable file:

#### **Steps**
1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```
2. Navigate to the directory containing `FINAL-BEST-GUI.py`.
3. Run PyInstaller:
   ```bash
   pyinstaller --onefile --noconsole CNN_GUI.py
   ```
4. After completion:
   - The `.exe` file will be located in the `dist` folder within your project directory.
5. Share the `.exe` file with users who do not have Python installed.

---

### **Notes**
- Ensure all dependencies (`tensorflow`, `numpy`, etc.) are installed before running or converting to `.exe`.
- For large datasets, ensure sufficient system memory and processing power are available during training.

Enjoy using this intuitive GUI-based CNN training and prediction software!
