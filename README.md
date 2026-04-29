## E-Commerce Furniture Sales Prediction — ML Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat&logo=scikit-learn)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange?style=flat&logo=mysql)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?style=flat&logo=pandas)
![Joblib](https://img.shields.io/badge/Model-Joblib%20Serialized-yellow?style=flat)
![TF-IDF](https://img.shields.io/badge/NLP-TF--IDF%20(1--2%20grams)-9cf?style=flat)

An end-to-end **machine learning pipeline** for predicting furniture product sales on e-commerce platforms. Built on 2,000 AliExpress furniture listings, the app combines **NLP (TF-IDF on product titles)**, **feature engineering**, **GridSearchCV hyperparameter tuning**, and a fully interactive **Streamlit dashboard** with MySQL integration and Excel export.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Feature Engineering](#feature-engineering)
- [Model Accuracy Results](#model-accuracy-results)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [References](#references)

## Project Overview
Predicting how many units of a furniture product will sell is challenging — it depends on price, discount strategy, shipping tags, and even the wording of the product title. This project addresses this by:
- Cleaning raw price strings (`$1,299.00` → `1299.0`) and engineering a `discount_percentage` feature
- Applying **TF-IDF (1–2 gram)** vectorisation on `productTitle` to extract text-based sales signals
- Encoding top shipping tags (`tagText`) using **OneHotEncoder**
- Training two `sklearn` Pipelines: **Linear Regression** and **Random Forest with GridSearchCV**
- Using **log1p / expm1 transformation** on the target (`sold`) to handle heavy skewness
- Saving trained pipelines as `.joblib` files for instant reload
- Pushing metrics and predictions to **MySQL** and exporting to **Excel**
- Serving everything through a multi-section **Streamlit app**

## Dataset
| Property | Detail |
| :--- | :--- |
| **File** | `ecommerce_furniture_dataset_2024.csv` |
| **Source** | AliExpress furniture listings (scraped) |
| **Rows** | 2,000 products |
| **Target Variable** | `sold` — number of units sold |
| **Target Transform** | `y_log = np.log1p(sold)` during training; `np.expm1()` on output |

### Original Columns
| Column | Type | Description |
| :--- | :---: | :--- |
| `productTitle` | Text | Product name and description |
| `price` | String → Float | Final selling price (cleaned from `$X,XXX`) |
| `originalPrice` | String → Float | Listed price before discount |
| `sold` | Numeric | Units sold — **prediction target** |
| `tagText` | Categorical | Shipping/extra tag (e.g., Free shipping) |

## Technologies Used
| Technology | Version | Purpose |
| :--- | :---: | :--- |
| **Python** | 3.9+ | Core programming language |
| **Pandas** | 2.x | Data loading, cleaning, feature engineering |
| **NumPy** | latest | Log transform, array operations |
| **Scikit-Learn** | 1.x | Pipelines, ColumnTransformer, TF-IDF, OHE, models, GridSearchCV, learning curves |
| **Joblib** | latest | Serialise and load trained pipelines |
| **Matplotlib** | latest | Learning curves, GridSearch plot, feature importance chart |
| **SQLAlchemy** | latest | MySQL ORM — write metrics and predictions |
| **mysql-connector-python** | latest | MySQL driver for SQLAlchemy |
| **Streamlit** | 1.x | Interactive ML pipeline dashboard |
| **OpenPyXL** | latest | Excel export via `pd.ExcelWriter` |
| **urllib.parse** | built-in | URL-encode MySQL password |
| **datetime / os** | built-in | Timestamped output files, directory creation |

### Python Libraries (from source code)
```python
import os, urllib.parse, joblib, numpy as np, pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine, text
```

## ML Pipeline Architecture
```
Raw CSV (2,000 AliExpress furniture products)
              │
              ▼
Data Cleaning
├── price / originalPrice: remove $, commas → float
├── sold → int (fill NaN with 0)
├── productTitle / tagText → fill NaN with "" / "unknown"
└── discount_percentage = (originalPrice - price) / originalPrice × 100

              │
              ▼
Feature Engineering
├── Top 10 tagText values → tagText_grouped (rest → "others")
└── y_log = np.log1p(sold)

              │
              ▼
ColumnTransformer (Preprocessor)
├── num    → passthrough  ["price", "discount_percentage"]
├── cat    → OneHotEncoder(handle_unknown="ignore")  ["tagText_grouped"]
└── text   → TfidfVectorizer(max_features=100, ngram_range=(1,2))  "productTitle"

              │
              ▼
Train / Test Split (80% / 20%, random_state=42)

              │
              ├─────────────────────────────────────┐
              ▼                                     ▼
Pipeline 1: LinearRegression          Pipeline 2: RandomForest + GridSearchCV
              │                          param_grid:
              │                          ├── n_estimators: [50]
              │                          ├── max_depth: [10, None]
              │                          └── min_samples_split: [2, 5]
              │                          cv=2, scoring=neg_MSE
              │                                     │
              └──────────────┬──────────────────────┘
                             │
                             ▼
              Predict → np.expm1(y_pred) → evaluate vs y_test_orig
                             │
                             ▼
              Save: linreg_pipeline.joblib + rf_pipeline.joblib
                             │
                             ▼
              Plots: learning curves, GridSearchCV results, feature importances
                             │
                             ▼
              Export: Excel (.xlsx) + MySQL (model_metrics, model_predictions)
```

---

## Feature Engineering
### Price Cleaning (from `clean_price_series()`)
```python
# Removes $, commas, strips whitespace, converts to float
s = series.astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
s = pd.to_numeric(s.replace({'': np.nan, 'nan': np.nan}), errors='coerce')
```
### Discount Percentage
```python
discount_percentage = np.where(
    originalPrice.notna() & (originalPrice > 0),
    ((originalPrice - price) / originalPrice) * 100,
    0.0
)
```
### Tag Grouping
```python
top_tags = df["tagText"].value_counts().nlargest(10).index
df["tagText_grouped"] = df["tagText"].apply(lambda x: x if x in top_tags else "others")
```
### 12 Features Fed to Models
| Feature | Type | Source |
| :--- | :---: | :--- |
| `price` | Numeric | Cleaned price column |
| `discount_percentage` | Numeric | Engineered from originalPrice |
| `tagText_grouped` (×11) | OHE | Top 10 tags + "others" |
| `productTitle` TF-IDF (×100) | Text | 1–2 gram TF-IDF |

## 📊 Model Accuracy Results
Both models trained on **80/20 split** (`random_state=42`). Target: `log1p(sold)`. Predictions inverse-transformed with `expm1()`.
### Model Performance
| Model | MSE | R² Score | Notes |
| :--- | :---: | :---: | :--- |
| **Linear Regression** | 5,398.36 | 0.016 | Baseline — weak fit due to non-linear sales patterns |
| **Random Forest** | 13,622.44 | -1.484 | GridSearchCV tuned — overfitting on sparse TF-IDF |
> **Note:** Low R² scores are expected here — predicting exact units sold from product metadata alone is a very hard regression problem. The value of this project is the **end-to-end ML pipeline architecture** (TF-IDF + OHE + ColumnTransformer + GridSearchCV + Streamlit), not the raw accuracy metric.

### GridSearchCV Parameters (Random Forest)
```python
param_grid = {
    "regressor__n_estimators": [50],
    "regressor__max_depth": [10, None],
    "regressor__min_samples_split": [2, 5],
}
# cv=2, scoring="neg_mean_squared_error", n_jobs=-1
```
### Saved Artifacts
| File | Contents |
| :--- | :--- |
| `output/linreg_pipeline.joblib` | Full trained Linear Regression pipeline |
| `output/rf_pipeline.joblib` | Full trained Random Forest pipeline (best GridSearch estimator) |
| `output/rf_feature_importances.csv` | Top 20 feature importances CSV |
| `output/plots/learning_curve_linreg.png` | Linear Regression learning curve |
| `output/plots/learning_curve_rf.png` | Random Forest learning curve |
| `output/plots/rf_gridsearch_results.png` | GridSearch CV MSE vs n_estimators |
| `output/plots/rf_feature_importance.png` | Top 20 feature importance bar chart |

## Streamlit Dashboard
### Sidebar Controls
| Control | Type | Purpose |
| :--- | :---: | :--- |
| Upload CSV | File uploader | Use custom dataset or fall back to default |
| MySQL credentials | Text inputs | User, password, host, port, database name |
| TF-IDF max features | Number input | Default: 100, step: 50 |
| Train models | Button | Runs full GridSearchCV training pipeline |
| Load saved pipelines | Button | Loads `.joblib` files from `output/` folder |
| Export to Excel | Button | Saves timestamped `.xlsx` with metrics + predictions |
| Push to MySQL | Button | Writes to `model_metrics` and `model_predictions` tables |

### Main Area Sections
| Section | Content |
| :--- | :--- |
| **Dataset Preview** | Row/column count + first 6 rows |
| **Data Cleaning Steps** | Expandable code + description of cleaning applied |
| **Cleaned Data Sample** | 6-row preview of cleaned columns |
| **Metrics Table** | MSE and R² for both models on test set |
| **Predictions Sample** | First 10 rows: true sold vs predicted (both models) |
| **Learning Curve — Linear** | Train vs Val MSE across training sizes |
| **Learning Curve — RF** | Train vs Val MSE across training sizes |
| **GridSearchCV Plot** | n_estimators vs mean CV MSE |
| **Feature Importances** | Top 20 RF features as horizontal bar chart + DataFrame |
| **Artifacts Footer** | Download buttons for `.joblib` and metrics CSV |

### MySQL Tables Created
| Table | Contents |
| :--- | :--- |
| `model_metrics` | MSE, RMSE, R², model name, timestamp |
| `model_predictions` | productTitle, true_sold, predicted_linreg, predicted_rf, price, discount, tag |

## Installation and Setup
### Step 1 — Clone the Repository
```bash
git clone https://github.com/abhi-1009/E-commerce-Furniture-Dataset-2024.git
cd E-commerce-Furniture-Dataset-2024
```
### Step 2 — Install Required Libraries
```bash
pip install streamlit pandas numpy scikit-learn joblib matplotlib sqlalchemy mysql-connector-python openpyxl
```
### Step 3 — Add the Dataset
Place `ecommerce_furniture_dataset_2024.csv` in the project folder and update:
```python
DEFAULT_CSV = "ecommerce_furniture_dataset_2024.csv"
```
### Step 4 — Configure MySQL (optional)
Enter credentials in the Streamlit sidebar at runtime — no hardcoding needed. The app will auto-create the database if it doesn't exist.

### Step 5 — Launch the App
```bash
streamlit run ecommerce_furniture_app.py
```
## Usage
1. **Launch the app** → dataset loads and preview appears automatically
2. **Sidebar** → optionally upload a custom CSV; enter MySQL credentials if needed
3. Click **Train models** → GridSearchCV runs, metrics and plots appear
4. Click **Load saved pipelines** → reload previously trained models instantly
5. Click **Export to Excel** → timestamped `.xlsx` downloaded with metrics + predictions
6. Click **Push to MySQL** → metrics and predictions written to MySQL database
7. **Artifacts footer** → download `.joblib` pipeline files directly from the app

## Key Insights
- **Discount percentage** is the strongest numeric predictor of units sold
- **TF-IDF product title features** capture keywords like "sofa", "chair", "set" that correlate with sales volume
- **Free shipping tag** (top `tagText` value) is a significant positive sales signal
- **Log transformation** on `sold` was essential — raw values are heavily right-skewed with extreme outliers
- **Low R²** reflects the inherent difficulty of predicting exact sales from metadata alone — additional features like ratings, reviews, and seller history would significantly improve accuracy

## References
- [Scikit-Learn Pipeline Documentation](https://scikit-learn.org/stable/modules/compose.html)
- [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Joblib Documentation](https://joblib.readthedocs.io/)

## Author
**Abhijit Sinha**
- GitHub: [@abhi-1009](https://github.com/abhi-1009)
- LinkedIn: [abhijit-sinha-053b159a](https://linkedin.com/in/abhijit-sinha-053b159a)
- Email: sinhaabhijit12@yahoo.com
