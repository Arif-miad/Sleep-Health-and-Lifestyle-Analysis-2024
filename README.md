

---

# Sleep Health and Lifestyle Analysis

This repository contains a project focused on analyzing the **Sleep Health and Lifestyle Dataset**. The dataset explores various lifestyle factors and their impact on sleep quality, with a focus on predicting sleep disorders such as **Insomnia** and **Sleep Apnea**.

## **Workflow Overview**
This project follows a structured approach to explore the dataset, perform data analysis, and build a classification model to predict sleep disorders.

### **Steps Followed in This Project:**
1. **Importing Libraries and Data**
   - Libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` are imported for data manipulation, analysis, and machine learning tasks.
   - The dataset is loaded and preprocessed for further analysis.

2. **Exploratory Data Analysis (EDA)**
   - Initial exploration and understanding of the dataset through summary statistics and visualizations.
   - Distribution of sleep disorders is analyzed to gain insights into the occurrence of different conditions.
   - Visualizations are used to analyze the correlation between different factors such as **Age**, **Gender**, **Occupation**, and **BMI** with sleep disorders.

3. **Feature Analysis**
   - **Gender**: Exploring how gender impacts sleep disorders.
   - **Occupation**: Analyzing the relationship between occupation and sleep patterns.
   - **BMI**: Analyzing how Body Mass Index (BMI) categories relate to sleep disorders like Insomnia and Sleep Apnea.
   - Additional analysis is done for other features like **Physical Activity Level**, **Stress Level**, and **Heart Rate**.

4. **Data Preprocessing**
   - Handling missing data (if any).
   - Encoding categorical features (e.g., Gender, Occupation, Sleep Disorder).
   - Scaling numerical values using **MinMaxScaler** for better model performance.

5. **Model Building: Logistic Regression**
   - The project uses **Logistic Regression** for classification to predict the **Sleep Disorder** column.
   - Training and testing data are prepared and split.
   - Model evaluation is performed using accuracy scores and confusion matrix.

## **Steps to Run the Project**
Follow these steps to run the code:

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/sleep-health-analysis.git
cd sleep-health-analysis
```

### **2. Install Required Libraries**
Create a virtual environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

### **3. Data Preprocessing**
The data is loaded and preprocessed. Missing values and categorical data are handled, and numerical values are scaled using **MinMaxScaler**.

```python
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
df = pd.read_csv('sleep_health_lifestyle.csv')

# Preprocessing (Handling Missing Data, Encoding Categorical Features)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Occupation'] = df['Occupation'].map({'Employed': 1, 'Unemployed': 0})
df['Sleep Disorder'] = df['Sleep Disorder'].map({'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2})

# Feature Scaling (MinMaxScaler)
scaler = MinMaxScaler()
numerical_cols = ['Age', 'Sleep Duration (hours)', 'Quality of Sleep (scale: 1-10)', 'Physical Activity Level (minutes/day)', 
                  'Stress Level (scale: 1-10)', 'Blood Pressure (systolic/diastolic)', 'Heart Rate (bpm)', 'Daily Steps']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

### **4. Exploratory Data Analysis (EDA)**
Visualize the distribution of the data, analyze key features, and check for correlations.

```python
# Visualizing distribution of Sleep Disorder
sns.countplot(x='Sleep Disorder', data=df)
plt.title('Distribution of Sleep Disorders')
plt.show()

# Correlation heatmap
corr_matrix = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

### **5. Logistic Regression Model**
Train and evaluate a **Logistic Regression** model for classification.

```python
# Splitting the data
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.show()
```

### **6. Results & Analysis**
- The Logistic Regression model achieves an accuracy of approximately **XX%**.
- The confusion matrix helps us understand how well the model is performing across each class (None, Insomnia, Sleep Apnea).

---

## **Conclusion**
This project analyzes the relationship between sleep health, lifestyle habits, and various cardiovascular metrics. Using **Logistic Regression**, we were able to classify sleep disorders with reasonable accuracy. Future improvements could involve experimenting with other models like **Random Forest** or **XGBoost**, adding feature engineering, and optimizing hyperparameters.

---

## **File Structure**
```
/sleep-health-analysis
    ├── sleep_health_lifestyle.csv          # Dataset file
    ├── analysis_notebook.ipynb            # Jupyter notebook with EDA and model code
    ├── requirements.txt                  # List of Python dependencies
    └── README.md                         # Project documentation
```

## **Acknowledgments**
This dataset is synthetic and was created to illustrate data analysis and machine learning techniques related to sleep health and lifestyle factors. 

