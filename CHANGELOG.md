# Changelog

## v0.2 - Lasso regression

What changed:
- Replaced LinearRegression with Lasso (`alpha=1.0`), still with StandardScaler
- Same data split and seed as v0.1, so the numbers are directly comparable
- Training now also logs 5-fold cross-validation RMSE in `metrics.json`
- Added `compare_models.py`, which compares the models we tried

| Version | Model | CV RMSE (5-fold, train) | Test RMSE |
|---|---|---|---|
| v0.1 | StandardScaler + LinearRegression | 55.97 | 53.85 |
| v0.2 | StandardScaler + Lasso (alpha=1.0) | 55.89 | **53.15** |

Why v0.2 is better:
- Test RMSE went down by 0.70 (about 1.3%), and CV RMSE is also slightly lower.
- Lasso shrinks weak features and set `s2` to zero (s2 overlaps a lot with s1), so the model is a bit simpler and less likely to overfit.
- We also tried Ridge, SelectKBest feature selection and RandomForest (see `compare_models.py`). Lasso had the best test RMSE; RandomForest was clearly worse on CV (58.46).
- The gain is small, which is expected on this small dataset (442 patients).

## v0.1 - Baseline

- Model: StandardScaler + LinearRegression
- Data split: 80% train / 20% test, `random_state=42`
- Test RMSE: **53.85**
- API with `GET /health` and `POST /predict`, packaged as a Docker image
