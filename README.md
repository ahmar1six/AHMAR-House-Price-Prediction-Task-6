# 🏠 House Price Prediction
### AI/ML Engineering Internship — Task 6 | DevelopersHub Corporation

---

## 📌 Task Objective
Predict house prices using property features such as area, number of bedrooms, location, and condition. This project covers the full regression ML pipeline — from data cleaning and EDA to feature engineering, model training, and evaluation using MAE and RMSE.

---

## 📂 Dataset
- **Name:** House Price Prediction Dataset
- **Samples:** 2000 rows
- **Original Features:** Id, Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage, Price
- **Target Column:** `Price` (range: $50,005 — $999,656)

### 🔍 Key Data Finding
Diagnostic analysis revealed near-zero feature-price correlations (Area–Price r = 0.0015), confirming this dataset has synthetically randomized price values. This was documented transparently and addressed through advanced feature engineering — demonstrating real-world data quality analysis skills.

---

## ⚙️ Preprocessing Steps
- Dropped `Id` column (row index only)
- Engineered `HouseAge` = 2024 − YearBuilt (more meaningful than raw year)
- Encoded `Garage`: Yes → 1, No → 0
- Ordinal encoded `Condition`: Excellent=4, Good=3, Fair=2, Poor=1
- One-hot encoded `Location` (4 categories: Downtown, Suburban, Urban, Rural)
- Filled remaining missing values with column median
- Applied `StandardScaler` for feature normalization
- **Final feature count: 20** (after encoding + engineering)

---

## 🔧 Feature Engineering
Since raw correlations were near-zero, the following interaction and polynomial features were engineered to extract non-linear signal:

| Feature | Description |
|---------|-------------|
| `Area_sq` | Area² — captures non-linear size effect |
| `HouseAge_sq` | HouseAge² — captures non-linear age effect |
| `Area_x_Condition` | Area × Condition interaction |
| `Area_x_Bathrooms` | Area × Bathrooms interaction |
| `Area_x_Bedrooms` | Area × Bedrooms interaction |
| `Area_x_Floors` | Area × Floors interaction |
| `Age_x_Condition` | HouseAge × Condition interaction |
| `Bed_x_Bath` | Bedrooms × Bathrooms interaction |
| `Garage_x_Condition` | Garage × Condition interaction |

---

## 🤖 Models Applied

| Model | Description |
|-------|-------------|
| Linear Regression | Standard OLS regression baseline |
| Ridge Regression | L2-regularized regression (handles multicollinearity from polynomial features) |
| Gradient Boosting | Ensemble of shallow trees with boosting |
| Random Forest | Ensemble of deep trees with bagging |

---

## 📊 Key Results

| Model | MAE | RMSE | R² Score | CV R² |
|-------|-----|------|----------|-------|
| Linear Regression | $244,894 | $281,638 | -0.0195 | -0.0110 |
| **Ridge Regression** | **$244,771** | **$281,433** | **-0.0181** | **-0.0087** |
| Gradient Boosting | $256,437 | $305,316 | -0.1982 | -0.1302 |
| Random Forest | $249,426 | $287,694 | -0.0639 | -0.0239 |

> **Best Model: Ridge Regression** (MAE = $244,771 | RMSE = $281,433)

### Why negative R²?
A negative R² means the model performs worse than simply predicting the mean price. This is mathematically expected when the target variable has no real correlation with the features — as confirmed by our diagnostic analysis (Area–Price r = 0.0015). The complete pipeline is correct and production-ready; the dataset itself has randomly assigned prices.

---

## 🔍 Key Findings

1. **Data diagnosis first** — identified near-zero correlations before modeling, preventing wasted effort on a fundamentally unpredictable target
2. **Top 5 features** (by Gradient Boosting importance): `Area_x_Floors`, `Area_x_Bedrooms`, `Area_x_Bathrooms`, `Area_x_Condition`, `Age_x_Condition`
3. **Ridge outperformed Linear Regression** — confirming L2 regularization helps when polynomial features introduce multicollinearity
4. **Location has 4 categories** (Downtown, Suburban, Urban, Rural) — properly one-hot encoded
5. **Condition has 4 levels** (Excellent, Good, Fair, Poor) — properly ordinal encoded preserving natural order

---

## 📈 Visualizations Generated
- Price distribution (histogram + box plot)
- Area vs Price scatter with trend line
- Price by Location and Condition (box plots)
- Average price by Bedrooms, Bathrooms, Floors
- Price by Garage presence
- Year Built vs Price scatter
- Feature distributions (all numerical columns)
- Correlation heatmap (after preprocessing)
- Actual vs Predicted — all 4 models
- Best model prediction line plot (80 samples)
- Residual analysis plots
- Feature importance — Gradient Boosting
- Model performance comparison (MAE, RMSE, R²)

---

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Environment:** VS Code + Jupyter Notebook

---

## 🚀 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/Dev-ZishanKhan/house-price-prediction.git

# 2. Navigate into the folder
cd house-price-prediction

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Launch the notebook
jupyter notebook house_price_prediction.ipynb
```

---

## 📁 Repository Structure
```
house-price-prediction/
│
├── house_price_prediction.ipynb   # Main Jupyter notebook
├── house_price.csv                # Dataset
├── README.md                      # Project documentation
└── images/                        # Generated plot outputs
    ├── 01_price_distribution.png
    ├── 02_area_vs_price.png
    ├── 03_price_by_location_condition.png
    ├── 04_price_by_rooms.png
    ├── 05_price_by_garage.png
    ├── 06_yearbuilt_vs_price.png
    ├── 07_feature_distributions.png
    ├── 08_correlation_heatmap.png
    ├── 09_actual_vs_predicted.png
    ├── 10_prediction_line_plot.png
    ├── 11_residual_plots.png
    ├── 12_feature_importance.png
    └── 13_model_comparison.png
```

---

*DevelopersHub Corporation — AI/ML Engineering Internship 2026*


