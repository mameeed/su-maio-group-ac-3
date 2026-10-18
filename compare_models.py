# Compares candidate models for v0.2 on the same split as train.py.
# Run: python compare_models.py
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_diabetes(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LinearRegression (v0.1)": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge alpha=10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "Lasso alpha=1 (v0.2)": make_pipeline(StandardScaler(), Lasso(alpha=1.0)),
    "SelectKBest k=6 + LR": make_pipeline(StandardScaler(), SelectKBest(f_regression, k=6), LinearRegression()),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
}

for name, model in models.items():
    cv_rmse = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error").mean()
    model.fit(X_train, y_train)
    rmse = mean_squared_error(y_test, model.predict(X_test)) ** 0.5
    print(f"{name:25} CV RMSE {cv_rmse:6.2f}   test RMSE {rmse:6.2f}")
