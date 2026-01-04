# Hiring Metrics Predictor 🧠📊

## Overview
Hiring Metrics Predictor is a machine learning project that predicts hiring decisions based on candidate attributes. The project uses a **custom synthetic dataset** generated with Faker to simulate real-world recruitment scenarios and compares multiple classification models to evaluate performance.

This project demonstrates an **end-to-end machine learning pipeline**, from data generation and preprocessing to model training, evaluation, and visualization.

---

## Problem Statement
Hiring decisions depend on multiple factors such as education, experience, technical skills, and certifications.  
The goal of this project is to analyze these factors and predict whether a candidate is likely to be **selected or not selected** using machine learning models.

---

## Dataset
- Custom synthetic dataset generated using **Faker**
- **500 candidate records**
- Features include:
  - Candidate Name
  - Degree
  - GPA
  - Technical Test Score
  - Number of Projects
  - English Proficiency
  - Years of Experience
  - Certifications
  - Internship Experience
  - Hackathon Participation
  - Age
- Target Variable:
  - **Selected** (Yes / No)

The dataset was created to closely resemble real recruitment data while avoiding privacy concerns.

---

## Tools & Technologies
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  

---

## Data Preprocessing
- Handling categorical variables using **One-Hot Encoding**
- Feature scaling using **StandardScaler**
- Train-test split for model evaluation
- Data cleaning and preparation

---

## Machine Learning Models Implemented
- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Gradient Boosting  
- XGBoost  

---

## Evaluation Metrics
The models were evaluated using:
- Accuracy Score  
- Classification Report (Precision, Recall, F1-score)  
- Confusion Matrix  
- ROC–AUC Score  

---

## Results & Visualization
- Model performances were compared using **bar graphs**
- Accuracy and ROC–AUC scores were visualized for all implemented models
- **Gradient Boosting and Random Forest models showed the strongest performance**, providing more accurate and reliable predictions compared to other classifiers
- These models were better at capturing complex patterns in the data, leading to improved hiring decision predictions

---

## Key Learnings
- Built a complete machine learning workflow from scratch
- Gained experience working with synthetic datasets
- Learned how different classification models perform on the same data
- Improved understanding of feature encoding, scaling, and evaluation metrics
- Strengthened skills in model comparison and result interpretation

---

## Future Improvements
- Hyperparameter tuning for improved model performance
- Feature importance analysis
- Testing the models on real-world recruitment datasets
- Deployment of the model as a web application

---

## Author
**Sanvi Sharma**  
- GitHub: https://github.com/sanvisharma29
- 📧 Email: sanvisharma592@gmail.com 
- 🔗 LinkedIn: www.linkedin.com/in/sanvisharma29
