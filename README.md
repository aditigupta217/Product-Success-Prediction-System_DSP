# 🏪 Product Success Prediction System

> A Machine Learning project that helps shopkeepers predict whether a product will **Succeed ✅** or **Fail ❌** before stocking it — using only pre-launch information.

---

## 📌 Project Overview

Stocking a shop is risky. Some products sell fast, others gather dust and waste money. This project uses **Logistic Regression** to help shopkeepers make smarter stocking decisions by predicting product success or failure based on 5 simple inputs — before spending any money.

**From a shopkeeper's perspective:**
> "Should I stock this product or not?"

---

## 🎯 Business Problem

A shopkeeper wants to decide **before buying a product** whether it will sell successfully in their store. They only know basic information at that point:

| Input | Example |
|---|---|
| Product Category | Electronics, Food, Clothing... |
| Seasonality | Winter, Summer, Spring, Autumn |
| Region | North, South, East, West |
| Price | $49.99 |
| Discount | 20% |

Based on these 5 inputs the model predicts → **Success (1)** or **Failure (0)**

---

## 📁 Project Structure

```
product-success-prediction/
│
├── notebook.ipynb                  ← Full ML training pipeline
├── app.py                          ← Flask backend
├── templates/
│   └── index.html                  ← HTML/CSS frontend
│
├── product_success_dataset.csv     ← Dataset (1000 rows)
│
├── product_success_model.pkl       ← Trained Logistic Regression model
├── scaler.pkl                      ← StandardScaler for Price and Discount
├── model_columns.pkl               ← Exact feature list from training
├── category_values.pkl             ← Dropdown values for web app
│
└── README.md                       ← This file
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Total Rows | 1000 |
| Total Columns | 6 (5 features + 1 target) |
| Target Column | Success (0 = Fail, 1 = Success) |
| Success Rate | 53.4% |
| Failure Rate | 46.6% |

### Features Used

| Column | Type | Description |
|---|---|---|
| Category | Categorical | Electronics, Clothing, Food, Toys, Furniture, Sports, Beauty, Books |
| Seasonality | Categorical | Spring, Summer, Autumn, Winter |
| Region | Categorical | North, South, East, West |
| Price | Numeric | Retail price in USD |
| Discount | Numeric | Discount percentage (0% to 50%) |
| **Success** | **Target** | **0 = Fail, 1 = Success** |

### ❌ Why Other Columns Were NOT Used

| Column | Reason Removed |
|---|---|
| Units Sold | Only known after launch — data leakage |
| Units Ordered | Post-launch operational data |
| Inventory Level | Changes after sales begin |
| Revenue | Result of success, not a cause |
| Competitor Pricing | Real-time data, not available pre-launch |

---

## 🤖 Machine Learning Pipeline

### Step 1 — One Hot Encoding
Converts text categories to numbers:
```
Category = "Electronics"
→ Category_Electronics = 1
→ Category_Clothing    = 0
→ Category_Food        = 0
```

### Step 2 — Outlier Treatment
IQR method used to detect and cap outliers in Price:
```
Q1 = $29.92  |  Q3 = $422.05  |  IQR = $392.13
Upper Bound = $1010.24
Outliers capped using Winsorization — no rows deleted
```

### Step 3 — StandardScaler
Brings Price and Discount to same scale:
```
Price    = $500  →  0.87  (scaled)
Discount = 10%   →  -0.32 (scaled)
```

### Step 4 — Train Test Split
```
Training set : 800 rows (80%)
Testing set  : 200 rows (20%)
stratify=y   : equal class distribution in both sets
```

### Step 5 — Logistic Regression
```python
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
```

### How Prediction Works
```
Shopkeeper enters 5 inputs
        ↓
One Hot Encoding → words become 0s and 1s
        ↓
StandardScaler → all numbers on same scale
        ↓
Weighted Sum → Score
        ↓
Sigmoid Function → Probability (0 to 1)
        ↓
Probability > 0.5 → SUCCESS ✅
Probability < 0.5 → FAILURE ❌
```

---

## 📈 Model Results

| Metric | Value |
|---|---|
| Testing Accuracy | ~65-68% |
| Precision (Success) | ~0.65 |
| Recall (Success) | ~0.67 |
| F1-Score | ~0.66 |
| AUC Score | ~0.70 |

---

## 🌐 Web Application

A Flask web app with pure HTML/CSS frontend allows shopkeepers to get instant predictions.

**Flow:**
```
Shopkeeper fills form → Flask receives inputs
→ Model predicts → Result shown with probability %
```

---

## 🚀 How to Run

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/product-success-prediction.git
cd product-success-prediction
```

### Step 2 — Install Required Libraries
```bash
pip install flask pandas numpy scikit-learn joblib matplotlib seaborn scipy
```

### Step 3 — Run the Notebook Once
Open `notebook.ipynb` in VS Code and run all cells top to bottom.
This trains the model and saves the 4 `.pkl` files automatically.

### Step 4 — Start the Web App
```bash
python app.py
```

### Step 5 — Open Browser
```
http://localhost:5000
```

---

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib + Seaborn | Plots and visualizations |
| Scikit-learn | ML model and preprocessing |
| Joblib | Save and load model files |
| Flask | Web backend server |
| HTML + CSS | Frontend user interface |



---
