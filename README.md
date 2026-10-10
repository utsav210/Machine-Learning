# 🧠 Machine Learning Training Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

> An enterprise-grade repository showcasing robust machine learning methodologies, focusing on predictive modeling for both classification and regression tasks.

## 📑 Table of Contents
- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Implemented Models](#-implemented-models)
  - [1. Credit Card Fraud Detection](#1-credit-card-fraud-detection-classification)
  - [2. Housing Price Prediction](#2-housing-price-prediction-regression)
  - [3. Medical Insurance Cost Prediction](#3-medical-insurance-cost-prediction-regression)
- [Documentation](#-documentation)
- [Technologies & Libraries](#-technologies--libraries)
- [Getting Started](#-getting-started)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Overview
This repository contains a collection of machine learning projects developed during ML training. It demonstrates industry-standard practices in data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, hyperparameter tuning, and evaluation.

## 🏗️ Project Architecture
```text
ML Training/
│
├── Classification_Model.ipynb     # Credit card fraud detection modeling
├── Housing_Price_Prediction.ipynb # Real estate pricing regression models
├── insurance.ipynb                # Medical insurance cost prediction
├── Credit Card Fraud Detection.pdf # Documentation & methodology for fraud detection
├── HPrice_Regression_Models.pdf   # Documentation & methodology for price prediction
├── README.md                      # Project documentation (You are here)
└── LICENSE                        # MIT License
```

## 🚀 Implemented Models

### 1. Credit Card Fraud Detection (Classification)
- **File**: [`Classification_Model.ipynb`](./Classification_Model.ipynb)
- **Objective**: Develop a high-accuracy, low-latency classification model to identify fraudulent credit card transactions.
- **Key Features**: Transaction metadata (timestamp, amount), anonymized PCA features, and geographic data.
- **Target Variable**: `is_fraud` (Binary: 0 for legitimate, 1 for fraudulent)
- **Techniques Used**: Handling highly imbalanced datasets (SMOTE/undersampling), anomaly detection, and classification algorithms (Random Forest, XGBoost).

### 2. Housing Price Prediction (Regression)
- **File**: [`Housing_Price_Prediction.ipynb`](./Housing_Price_Prediction.ipynb)
- **Objective**: Predict real estate market values based on property specifications and historical data.
- **Key Features**: Square footage, number of bedrooms/bathrooms, location demographics, and building conditions.
- **Target Variable**: `price` (Continuous)
- **Techniques Used**: Multicollinearity handling, outlier removal, feature scaling, and advanced regression algorithms (Linear Regression, Gradient Boosting).

### 3. Medical Insurance Cost Prediction (Regression)
- **File**: [`insurance.ipynb`](./insurance.ipynb)
- **Objective**: Estimate personal medical insurance charges based on individual health and demographic factors.
- **Key Features**: Age, BMI, smoking status, region, and number of dependents.
- **Target Variable**: `charges` (Continuous)
- **Techniques Used**: Categorical encoding, data normalization, and predictive modeling for health-related cost analysis.

## 📚 Documentation
Comprehensive documentation detailing the methodologies, mathematical foundations, and business impact for these models is provided:
- 📄 **[Credit Card Fraud Detection Documentation](./Credit%20Card%20Fraud%20Detection.pdf)**
- 📄 **[Housing Price Prediction Documentation](./HPrice_Regression_Models.pdf)**

## 🛠️ Technologies & Libraries
The projects leverage the modern Python data science ecosystem:
- **Core**: `Python 3.8+`
- **Data Manipulation**: `Pandas`, `NumPy`
- **Machine Learning**: `Scikit-Learn`
- **Data Visualization**: `Matplotlib`, `Seaborn`
- **Environment**: `Jupyter Notebook` / `JupyterLab`

## 🏁 Getting Started

### Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/utsav210/Machine-Learning.git
   cd Machine-Learning
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn seaborn jupyter
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Run Notebooks**
   Open any of the `.ipynb` files and execute the cells sequentially to reproduce the models and analyses.

## 📄 License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 📬 Contact
**Utsav**  
Email: [utsavgandhi273@gmail.com](mailto:utsavgandhi273@gmail.com)  
GitHub: [utsav210](https://github.com/utsav210)

---
*Built with passion for Data Science & Machine Learning.*