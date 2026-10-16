# California Housing Price Prediction (Machine Learning Regression)

A machine learning project to predict median house values in California districts using `scikit-learn`, comparing multiple regression models.

## Dataset
- **Source:** Scikit-learn built-in dataset (`fetch_california_housing`)
- **Samples:** 20,640 rows
- **Features:** 8 numerical features (Median Income, House Age, Average Rooms, Average Bedrooms, Population, Average Occupancy, Latitude, Longitude)
- **Target:** Median House Value (`MedHouseVal`)

## Models Compared
1. **Linear Regression:** Baseline model ($R^2 = 0.58$, $MAE = 0.53$)
2. **Decision Tree Regressor:** Non-linear single tree ($R^2 = 0.62$, $MAE = 0.45$)
3. **Random Forest Regressor:** Ensemble of 100 decision trees ($R^2 = 0.81$, $MAE = 0.33$)

## Results & Conclusion
The **Random Forest Regressor** significantly outperformed the linear and single-tree models, achieving an $R^2$ score of **0.81** and lowering the Mean Absolute Error (MAE) to approximately **$33,000**.
