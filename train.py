import json
import os

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "v0.2"
SEED = 42

# load data (y = progression index, higher = worse)
data = load_diabetes(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]

# 80% train, 20% test, fixed seed so the split is always the same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# v0.2: Lasso instead of LinearRegression (shrinks weak features)
model = make_pipeline(StandardScaler(), Lasso(alpha=1.0))

# 5-fold cross-validation on the training data only
cv_rmse = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error").mean()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions) ** 0.5
print(f"Model {MODEL_VERSION} CV RMSE: {cv_rmse:.2f}, test RMSE: {rmse:.2f}")

os.makedirs("model", exist_ok=True)
joblib.dump({"model": model, "version": MODEL_VERSION}, "model/model.joblib")

with open("model/metrics.json", "w") as f:
    json.dump({"model_version": MODEL_VERSION, "cv_rmse": round(cv_rmse, 2), "rmse": round(rmse, 2)}, f, indent=2)
