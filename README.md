# 🧠 Machine Learning Training Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

> An enterprise-grade repository showcasing robust machine learning methodologies, focusing on predictive modeling for both classification and regression tasks. 

**Developed by a passionate Machine Learning Engineer dedicated to writing clean, scalable, and highly optimized ML pipelines.**

## 📑 Table of Contents
- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Implemented Models in Detail](#-implemented-models-in-detail)
  - [1. Credit Card Fraud Detection](#1-credit-card-fraud-detection-classification)
  - [2. Housing Price Prediction](#2-housing-price-prediction-regression)
  - [3. Medical Insurance Cost Prediction](#3-medical-insurance-cost-prediction-regression)
  - [4. Titanic Survival Prediction](#4-titanic-survival-prediction-classification)
  - [5. Hyperparameter Tuning (GridSearchCV)](#5-hyperparameter-tuning-gridsearchcv)
  - [6. Ensemble Learning Techniques](#6-ensemble-learning-techniques)
  - [7. Clustering Techniques (Unsupervised Learning)](#7-clustering-techniques-unsupervised-learning)
  - [8. Dimensionality Reduction (PCA)](#8-dimensionality-reduction-pca)
- [Documentation](#-documentation)
- [Technologies & Libraries](#-technologies--libraries)
- [Getting Started](#-getting-started)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Overview
This repository contains a comprehensive collection of machine learning projects developed to solve complex, real-world data problems. It demonstrates industry-standard practices across the entire machine learning lifecycle: data preprocessing, exploratory data analysis (EDA), advanced feature engineering, rigorous model selection, hyperparameter tuning, and robust evaluation metrics.

## 🏗️ Project Architecture
```text
ML Training/
│
├── Classification_Model.ipynb     # Credit card fraud detection modeling
├── Housing_Price_Prediction.ipynb # Real estate pricing regression models
├── insurance.ipynb                # Medical insurance cost prediction
├── Titanic_Survival_Prediction.ipynb # Passenger survival classification
├── GridSearchCV.ipynb             # Hyperparameter tuning tutorials
├── EnsembleLearning.ipynb         # Ensemble learning models (Random Forest, Boosting)
├── Clustering.ipynb               # Unsupervised learning (K-Means, Hierarchical)
├── PCADimensionalityReduction.ipynb # Principal Component Analysis techniques
├── Credit Card Fraud Detection.pdf # Documentation & methodology for fraud detection
├── HPrice_Regression_Models.pdf   # Documentation & methodology for price prediction
├── README.md                      # Project documentation (You are here)
└── LICENSE                        # MIT License
```

## 🚀 Implemented Models in Detail

### 1. Credit Card Fraud Detection (Classification)
- **File**: [`Classification_Model.ipynb`](./Classification_Model.ipynb)
- **Problem Statement**: Credit card fraud costs billions annually. Detecting fraudulent transactions in real-time with minimal false positives is critical for financial institutions.
- **Algorithms Used & Why**: 
  - **Random Forest**: Chosen for its robustness against overfitting and its innate ability to map complex, non-linear financial patterns without extensive scaling.
  - **XGBoost**: Employed for its gradient boosting framework which sequentially minimizes errors, capturing subtle anomalies that basic algorithms miss, ensuring high precision.
- **Workflow & Methodology**: 
  - Conducted extensive Exploratory Data Analysis (EDA) to understand the distribution of anonymized PCA features.
  - Addressed extreme class imbalance (a hallmark of fraud datasets) using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to prevent model bias towards majority class.
  - Evaluated the model strictly on precision, recall, and the F1-score (rather than generic accuracy) to ensure false negatives (undetected fraud) were absolutely minimized.

### 2. Housing Price Prediction (Regression)
- **File**: [`Housing_Price_Prediction.ipynb`](./Housing_Price_Prediction.ipynb)
- **Problem Statement**: Accurately predicting real estate values is vital for market analysis, property investment, and automated valuation models (AVMs).
- **Algorithms Used & Why**: 
  - **Multiple Linear Regression**: Serves as a strong, interpretable baseline model allowing for coefficient analysis (e.g., precisely determining the financial impact of adding a bathroom).
  - **Gradient Boosting Regressor**: Utilized to capture non-linear interactions between features like location demographics and building condition, yielding superior $R^2$ scores over linear baselines.
- **Workflow & Methodology**:
  - Engineered sophisticated features combining metrics (e.g., price per square foot) and algorithmically detected and removed outliers in housing metrics to stabilize the model.
  - Ensured multicollinearity was treated (using VIF) and scaled features for algorithms sensitive to variance.
  - Validated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) for tangible financial interpretability.

### 3. Medical Insurance Cost Prediction (Regression)
- **File**: [`insurance.ipynb`](./insurance.ipynb)
- **Problem Statement**: Estimating medical insurance charges based on personal health data allows insurance companies to formulate risk assessments and fair premium pricing structures.
- **Algorithms Used & Why**: 
  - **Decision Trees & Random Forest Regressor**: Selected to effortlessly model non-linear segments (e.g., capturing the massive, distinct cost jumps for smokers vs. non-smokers).
  - **Polynomial Regression**: Explored to mathematically model exponential and compounded relationships between variables like age/BMI and subsequent medical costs.
- **Workflow & Methodology**:
  - Executed One-Hot encoding for categorical variables like `sex`, `smoker`, and `region`.
  - Analyzed the Pearson correlation matrix to discover that `smoker` status and `BMI` are the dominating predictors of high charges.
  - Tuned the models to generalize well across unseen data without overfitting on localized regional data.

### 4. Titanic Survival Prediction (Classification)
- **File**: [`Titanic_Survival_Prediction.ipynb`](./Titanic_Survival_Prediction.ipynb)
- **Problem Statement**: A cornerstone ML classification challenge aimed at predicting passenger survival probabilities based on socio-economic status, demographics, and cabin layouts.
- **Algorithms Used & Why**: 
  - **Logistic Regression**: Employed as an inferential algorithm to cleanly understand the log-odds of survival based on gender and ticket class.
  - **Support Vector Machines (SVM)**: Used to compute the optimal hyperplane that maximizes the margin separating survivors and non-survivors in a high-dimensional feature space.
- **Workflow & Methodology**:
  - Imputed missing `Age` values intelligently using median estimations grouped dynamically by passenger class and gender.
  - Performed string manipulation to extract titles from names (e.g., Mr., Mrs., Miss) creating a highly predictive new feature capturing social status.
  - Cleaned data pipelines by dropping non-informative columns (`Cabin`, `Ticket`) to drastically reduce statistical noise.

### 5. Hyperparameter Tuning (GridSearchCV)
- **File**: [`GridSearchCV.ipynb`](./GridSearchCV.ipynb)
- **Objective**: Moving beyond default, out-of-the-box parameters to systematically squeeze maximum performance and accuracy out of deployed machine learning models.
- **Workflow & Methodology**: 
  - Implemented **GridSearchCV** (via Scikit-Learn) to exhaustively and automatically search through a multidimensional hyperparameter grid.
  - Integrated **K-Fold Cross-Validation** to strictly ensure the hyperparameter tuning didn't simply overfit to a specific training set split, guaranteeing generalizability.
  - Documented how tweaking parameters like `max_depth`, `n_estimators`, and `learning_rate` systematically shifted an estimator's bias-variance tradeoff.

### 6. Ensemble Learning Techniques
- **File**: [`EnsembleLearning.ipynb`](./EnsembleLearning.ipynb)
- **Objective**: Combining multiple independent weak learners into a single, highly robust predictive model to combat variance and bias.
- **Algorithms Explored**:
  - **Bagging (Random Forest)**: Constructed parallel decision trees on bootstrapped data subsets, aggregating their votes to drastically reduce model variance.
  - **Boosting (AdaBoost & Gradient Boosting)**: Constructed sequential trees where each new tree specifically targets and corrects the residual errors of the previous ones, successfully minimizing bias.
- **Workflow & Methodology**: 
  - Conducted head-to-head comparisons of individual base estimators against their ensemble counterparts to empirically prove the performance uplift.
  - Analyzed the trade-off between computational expense (training time) and predictive accuracy gains.

### 7. Clustering Techniques (Unsupervised Learning)
- **File**: [`Clustering.ipynb`](./Clustering.ipynb)
- **Problem Statement**: Discovering hidden structures, customer segments, and natural groupings in completely unlabelled data.
- **Algorithms Used & Why**: 
  - **K-Means Clustering**: Deployed for its speed, scalability, and efficiency in partitioning large datasets into distinct, non-overlapping subgroups based on centroids.
  - **Hierarchical Clustering (Agglomerative)**: Used to build a dendrogram, allowing stakeholders to visually interpret the data hierarchy and granularity without pre-specifying the number of clusters.
- **Workflow & Methodology**:
  - Applied the mathematical **Elbow Method** and calculated **Silhouette Scores** to deduce the most statistically sound number of clusters ($K$).
  - Standardized all numerical features beforehand, an absolutely vital step since distance-based clustering algorithms are highly sensitive to unscaled variations.

### 8. Dimensionality Reduction (PCA)
- **File**: [`PCADimensionalityReduction.ipynb`](./PCADimensionalityReduction.ipynb)
- **Problem Statement**: Overcoming the "curse of dimensionality" by mathematically compressing massive feature spaces while retaining the dataset's critical variance and signal.
- **Algorithms Used & Why**: 
  - **Principal Component Analysis (PCA)**: An orthogonal linear transformation used to project high-dimensional data into a lower-dimensional subspace, forcing the maximization of variance onto new independent axes.
- **Workflow & Methodology**:
  - Calculated the covariance matrix and extracted eigenvectors/eigenvalues from the scaled data.
  - Mapped the cumulative explained variance ratio to determine the absolute minimum number of principal components needed to retain 95%+ of the original data's underlying information.
  - Demonstrated the drastic reduction in model training time and memory complexity when pipelining PCA-transformed data versus raw high-dimensional data.

## 📚 Documentation
Comprehensive PDF documentation detailing the mathematical foundations, methodologies, and business impact for these models is provided:
- 📄 **[Credit Card Fraud Detection Documentation](./Credit%20Card%20Fraud%20Detection.pdf)**
- 📄 **[Housing Price Prediction Documentation](./HPrice_Regression_Models.pdf)**

## 🛠️ Technologies & Libraries
The projects leverage the modern Python data science ecosystem:
- **Core**: `Python 3.8+`
- **Data Manipulation**: `Pandas`, `NumPy`
- **Machine Learning**: `Scikit-Learn`, `XGBoost`
- **Data Visualization**: `Matplotlib`, `Seaborn`
- **Environment**: `Jupyter Notebook` / `JupyterLab`

## 🏁 Getting Started

### Prerequisites
Ensure you have Python installed. It is highly recommended to use a virtual environment (`venv` or `conda`).

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/utsav210/Machine-Learning.git
   cd Machine-Learning
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn seaborn xgboost jupyter
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Run Notebooks**
   Open any of the `.ipynb` files and execute the cells sequentially to reproduce the end-to-end ML pipelines.

## 📄 License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 📬 Contact
**Utsav**  
Machine Learning Engineer  
Email: [utsavgandhi273@gmail.com](mailto:utsavgandhi273@gmail.com)  
GitHub: [utsav210](https://github.com/utsav210)

---
*Built with passion for Data Science & Machine Learning. Actively developing and pushing boundaries.*