# House Price Prediction
### AI/ML Engineering Internship — Task 6 | DevelopersHub Corporation

---

# Task Objective
Predict house prices using property features such as area, number of bedrooms, location, and condition. This project covers the full regression ML pipeline — from data cleaning and EDA to feature engineering, model training, and evaluation using MAE and RMSE.

---

# Dataset
- **Name:** House Price Prediction Dataset
- **Samples:** 2000 rows
- **Original Features:** Id, Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage, Price
- **Target Column:** `Price` (range: $50,005 — $999,656)

#Key Data Finding
Diagnostic analysis revealed near-zero feature-price correlations (Area–Price r = 0.0015), confirming this dataset has synthetically randomized price values. This was documented transparently and addressed through advanced feature engineering — demonstrating real-world data quality analysis skills.

---

# Preprocessing Steps
- Dropped `Id` column (row index only)
- Engineered `HouseAge` = 2024 − YearBuilt (more meaningful than raw year)
- Encoded `Garage`: Yes → 1, No → 0
- Ordinal encoded `Condition`: Excellent=4, Good=3, Fair=2, Poor=1
- One-hot encoded `Location` (4 categories: Downtown, Suburban, Urban, Rural)
- Filled remaining missing values with column median
- Applied `StandardScaler` for feature normalization
- **Final feature count: 20** (after encoding + engineering)

---

##  Feature Engineering
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

##  Models Applied

| Model | Description |
|-------|-------------|
| Linear Regression | Standard OLS regression baseline |
| Ridge Regression | L2-regularized regression (handles multicollinearity from polynomial features) |
| Gradient Boosting | Ensemble of shallow trees with boosting |
| Random Forest | Ensemble of deep trees with bagging |

---

##  Key Results

| Model | MAE | RMSE | R² Score | CV R² |
|-------|-----|------|----------|-------|
| Linear Regression | $244,894 | $281,638 | -0.0195 | -0.0110 |
| **Ridge Regression** | **$244,771** | **$281,433** | **-0.0181** | **-0.0087** |
| Gradient Boosting | $256,437 | $305,316 | -0.1982 | -0.1302 |
| Random Forest | $249,426 | $287,694 | -0.0639 | -0.0239 |

> **Best Model: Ridge Regression** (MAE = $244,771 | RMSE = $281,433)


