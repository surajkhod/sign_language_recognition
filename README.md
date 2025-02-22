# 🖐️ American Sign Language Recognition 🤟  

Welcome to the **American Sign Language Recognition** project! This project recognizes hand signs from the **American Sign Language (ASL)** alphabet using a machine learning model trained with custom image data.  

---

## 📌 Features  
✅ **Custom Hand Sign Dataset** – Collected using a built-in script.  
✅ **Trained Deep Learning Model** – Uses Google Teachable Machine (`keras_model.h5`).  
✅ **Real-time Sign Recognition** – Displays recognized sign on screen.  
✅ **Speech Feature** – Speaks out the recognized sign when pressing `S` key.  

---

## 📂 Project Structure  
📁 Sign-Language-Recognition

│── 📁 Data/ # Contains images of hand signs (A-Z)

│── 📁 Train Model/ # Stores trained model (keras_model.h5)

│── 📝 data_collection.py # Script to collect images for dataset

│── 📝 test.py # Main script to run the recognition

│── 📝 speech.py # Experimental file for voice features

│── 📝 README.md # Project Documentation


---

## 🛠️ Installation & Setup  

### 📌 Prerequisites  
**🔹 Python Version:** This project requires **Python 3.8.10**.  
**🔹 Install Required Libraries:** Run the following command to install all dependencies:  
```sh
pip install opencv-python cvzone numpy pyttsx3
```
---

## ✅ Required Python Libraries  

Before running the project, ensure you have the following libraries installed. If not, install them using the provided command.

```sh
pip install opencv-python cvzone numpy pyttsx3
```
### 📌 Importing necessary libraries
```
import cv2  # OpenCV for image processing
from cvzone.HandTrackingModule import HandDetector  # Hand detection module
from cvzone.ClassificationModule import Classifier  # Classification module for model prediction
import numpy as np  # Numerical operations
import math  # Math operations
import time  # Time module for delay functions
import pyttsx3  # Text-to-speech conversion
```
---

## 🚀 How to Use  

### 1️⃣ Collect Your Own Data  

To collect images for training, run "data collection.py" file.
- A window will open showing a box on the screen.
- Place your hand inside the box and press "S" on your keyboard to capture an image.
- The number of collected images will be displayed in the terminal.
- The images will be stored in the Data/ folder inside their respective subfolders (A-Z).
- Recommendation: Collect at least 500+ images per letter for better accuracy.

### 2️⃣ Train the Model
- The model was trained using Google Teachable Machine.
- After collecting enough images, upload them to Google Teachable Machine and train the model.
- Export the trained model as a .h5 file.
- Move the generated keras_model.h5 file into the Train Model/ folder.

### 3️⃣ Run Sign Recognition
- Run test.py file to Start real-time recognition:
- Press "S" to hear the letter spoken aloud

### 3️⃣ Speech.py
speech.py is just file for try and error purpose, where you can experiment other things if you want, it has nothing to do with others.

## 📢 Contributing
Want to improve this project? Feel free to fork, improve, and submit a PR! 🚀
