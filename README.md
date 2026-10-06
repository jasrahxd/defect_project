# 🏭 AI-Based Industrial Defect Detection System

A high-performance machine vision system designed for factory assembly lines. This system analyzes product integrity metrics in real-time, isolating surface defects, structural fractures, and processing contamination using an unsupervised **Isolation Forest** anomaly framework.

## 🚀 System Architecture
- **Vision Preprocessing:** Automated resizing, grayscale matrices transformation, and contrast scaling via **OpenCV**.
- **Core Engine:** Statistical boundary isolation using **Scikit-Learn** to handle imbalanced production data.
- **Operator Dashboard:** Reactive real-time quality control control interface deployed via **Streamlit**.

## 🛠️ Step-by-Step Execution Guide

### 1. Build Isolated Environment
```bash
python -m venv myenv
.\myenv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Sample Factory Matrices
```bash
.\myenv\Scripts\python.exe src/generate_data.py
```

### 3. Initialize AI Training Core
```bash
.\myenv\Scripts\python.exe src/train.py
```

### 4. Deploy Factory Interface
```bash
.\myenv\Scripts\streamlit.exe run app.py
```
