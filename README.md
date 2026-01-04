# 🌽 Corn / Maize Leaf Disease Detection

An end-to-end **Machine Learning image classification project** that detects common diseases in **corn (maize) leaves** using image processing and a trained Decision Tree model.  
The application is deployed using **Streamlit** and **Hugging Face Spaces**.

---

## 🚀 Live Demo
🔗 https://huggingface.co/spaces/maheshdugyani/Corn_Leaf_Disease_Detection/corn-leaf-disease-detection

---

## 📌 Problem Statement
Farmers often struggle to identify crop diseases early, leading to reduced yield.  
This project aims to **automatically detect corn leaf diseases** from images, enabling faster diagnosis and better crop management.

---

## 🧠 Model Details
- **Model Used:** Decision Tree Classifier  
- **Input:** Corn leaf image (RGB)  
- **Image Size:** 300 × 300  
- **Classes:**
  - Common Rust
  - Gray Leaf Spot
  - Blight
  - Healthy

⚠️ *The model is trained only on corn leaf images. Predictions on non-leaf images may be unreliable.*

---

## 📊 Dataset Overview
| Class | Images |
|------|--------|
| Common Rust | 1306 |
| Gray Leaf Spot | 574 |
| Blight | 1146 |
| Healthy | 1162 |

---

## 🛠️ Tech Stack
- Python  
- NumPy  
- OpenCV  
- Scikit-learn  
- Streamlit  
- Hugging Face Spaces  

