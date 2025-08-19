
# Machine Learning – Bank Marketing Campaign Prediction

## 📌 Project Overview

This project applies machine learning techniques to predict whether a client will subscribe to a term deposit (yes/no) based on data from direct marketing campaigns of a Portuguese banking institution.

The campaigns were conducted via phone calls, and sometimes multiple contacts with the same client were required to determine the subscription outcome.

The project explores different datasets, preprocessing steps, and classification models to evaluate predictive performance.

---

**Target Variable:**

* `y`: whether the client subscribed to a term deposit (`yes`/`no`)

---

## 📈 Results & Insights

* The dataset is **imbalanced** (majority "no" responses).
* Baseline accuracy can be misleading → metrics like **ROC-AUC** and **F1-score** are more reliable.
* Ensemble methods (Random Forest, Gradient Boosting) outperform simple models.
