# 🏎️ TORCS Racing Car Simulator Client (MLP-Based ANN)

This project contains an intelligent autonomous driving client for the [TORCS (The Open Racing Car Simulator)](http://torcs.sourceforge.net/) simulator. The client uses an **Artificial Neural Network (ANN)**, specifically a **Multi-Layered Perceptron (MLP)**, to predict driving actions such as steering, acceleration, and braking based on telemetry data.

---

## 📌 Project Highlights

- 🧠 Built using **TensorFlow**, **PyTorch**, **Keras**, and **scikit-learn**.
- 🤖 Trained on **real-time telemetry data** from TORCS to predict driving decisions.
- 🗂️ Contains a **pretrained model** for demonstration purposes (trained on 50,000 telemetry points).
- 🏗️ Fully extensible for larger datasets and advanced neural network models.

---

## 📁 Repository Structure


---

## ⚙️ Technology Stack

- **Languages**: Python 3.8+
- **Frameworks/Libraries**:
  - `TensorFlow`
  - `Keras`
  - `PyTorch`
  - `scikit-learn`
  - `pandas`, `numpy`, `matplotlib` for data handling and visualization

---

## 📊 Dataset

- File: `telemetry_log.csv`
- Total Entries: **50,000 telemetry records**
- Features include:
  - Track position
  - Speed
  - RPM
  - Steering angle
  - Distance from track edge sensors
  - Acceleration, brake, and gear values

> 🧩 You can **improve accuracy** significantly by increasing the dataset size to **100,000+ lines**. This will help the MLP generalize better across diverse driving situations and track layouts.

---

## 🧠 Model Overview

- **Type**: Multi-Layered Perceptron (MLP)
- **Inputs**: Sensor values (track sensors, speed, RPM, etc.)
- **Outputs**: Steering angle, acceleration, and brake intensity
- **Training Details**:
  - Loss function: MSE (Mean Squared Error)
  - Optimizers used: Adam, SGD (based on framework)
  - Trained using `project.py` on `telemetrylog.csv`
  - Pretrained model saved in `src/torcs_driver_metadata.pkl`
---
## 🚀 How to Use

### 1. Install Dependencies

Make sure you have Python 3.8 installed. Do not use python 3.11+ because of Tensor Flow. Then install required packages:

```bash
pip install -r requirements.txt
```
Let me know if you also want a `requirements.txt` file auto-generated based on your dependencies, or badges for Python version, license, or build status.
