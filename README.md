---

Heart Disease Prediction using Supervised Learning

#This project explores **Supervised Learning** to predict heart disease from patient data using **Logistic Regression, k-NN, and Decision Trees**. Along the way, I learned how preprocessing, feature importance, and model selection dramatically affect performance — and yes, Lagos traffic counts as debugging practice 😅.

---

## **Why These Models?**

* **Logistic Regression:** Simple, interpretable, and perfect for understanding which features actually influence heart disease.
* **k-Nearest Neighbors (k-NN):** Intuitive and demonstrates how similar patients cluster together.
* **Decision Trees:** Visual, easy to interpret, and great for storytelling — I could literally “see” how decisions were made.

---

## **Project Insights**

### Logistic Regression (85% Accuracy)

* Clean and interpretable.
* Top predictors: chest pain type, ST depression (`oldpeak`), maximum heart rate (`thalch`), age, and cholesterol.
* Sex influences risk slightly (male higher, female lower).

### k-Nearest Neighbors (~80% Accuracy)

* Sensitive to feature scaling — standardization improves performance.
* Top features: cholesterol, chest pain type, ST depression.

### Decision Tree (~74% Accuracy)

* Visual and intuitive but prone to overfitting.
* Key features: asymptomatic chest pain, cholesterol, ST depression, max heart rate, age.

**Key Takeaways:** Across all models, the most influential features were chest pain type, cholesterol, ST depression, max heart rate, and age. Sex mattered mostly for Logistic Regression.

---

## **Installation**

1. Clone the repository:

```bash
!git clone https://github.com/Miideh/Heart-Disease-Prediction-Supervised-Learning-.git
cd heart-disease-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## **Usage: Run End-to-End**

### **Step 1: Prepare Data**

* Place your dataset CSV file in the `data/` folder (example: `data/heart.csv`).

### **Step 2: Train Models**

* Open `notebooks/Supervised Learning: Predicting Heart Disease.ipynb`.
* Run all cells to preprocess data, train models, evaluate performance, and visualize results.

### **Step 3: Evaluate Models**

* Get:

  * Accuracy and ROC AUC
  * Confusion matrices
  * Feature importance plots
  * Decision Tree visualization

### **Step 4: Save Trained Models**

* Models are saved in `models/`:

```python
import joblib
joblib.dump(fitted_models['LogisticRegression'], 'models/LogisticRegression_pipeline.pkl')
```

### **Step 5: Make Predictions on New Data**

```python
import joblib
import pandas as pd

pipeline = joblib.load('models/LogisticRegression_pipeline.pkl')
new_data = pd.read_csv('data/new_patients.csv')
predictions = pipeline.predict(new_data)
print(predictions)
```

---

## **Requirements**

```text
pandas==2.1.1
numpy==1.25.2
matplotlib==3.8.0
seaborn==0.12.3
scikit-learn==1.3.2
joblib==1.3.2
sqlite3
```

> **PS:** Pipelines handle feature scaling and encoding automatically. Always ensure new data matches the training schema.

---

## **Closing Thoughts**

Supervised learning taught me that understanding data isn’t just about crunching numbers, it’s about seeing **relationships** and **connections**. Logistic Regression strikes a balance between performance and interpretability, k-NN shows numeric similarity matters, and Decision Trees make storytelling easy.


---


