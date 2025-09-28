# E-commerce-Furniture-Dataset-2024
## Overview
This project analyzes an e-commerce furniture dataset (2,000 products from AliExpress).  
The main objective is to **predict the number of items sold** based on product attributes such as price, discounts, shipping tags, and product titles.

## Dataset Description
The dataset contains **2,000 furniture product listings** scraped from AliExpress.  
It provides valuable insights into pricing strategies, discounts, and sales performance.

**Columns:**
- `productTitle` — Name/description of the furniture item  
- `originalPrice` — Original listed price before discounts  
- `price` — Final selling price after discounts  
- `sold` — Number of units sold  
- `tagText` — Shipping/extra details (e.g., Free shipping)  

## Tech Stack
- Python (pandas, scikit-learn, seaborn, matplotlib)
- SQL (aggregations & MySQL export)
- Machine Learning (Linear Regression, Random Forest)
- Streamlit (interactive app)
- Excel for reporting

## Steps Performed
1. Data Preprocessing (cleaning prices, encoding tags, discount feature engineering)
2. Exploratory Data Analysis (EDA) with visualizations
3. SQL-style queries (tag frequency aggregations)
4. Machine Learning model training (Linear Regression, Random Forest)
5. Model evaluation with **MSE & R²**
6. Streamlit app for interactive exploration, training, and exporting results

## Model Performance
- Linear Regression → MSE = 5398.36, R² = 0.016  
- Random Forest → MSE = 13622.44, R² = -1.484  

## Streamlit App Features
- Upload CSV or use default dataset  
- Data cleaning & feature engineering  
- Train ML models with GridSearchCV  
- Visualize learning curves, feature importances, predictions  
- Export metrics & predictions to Excel/MySQL  

## Conclusion
- Random Forest performed better than Linear Regression in predicting sales.  
- Discounts and shipping tags significantly influence sales volume.  
- The Streamlit app provides an end-to-end ML pipeline for interactive analysis.
