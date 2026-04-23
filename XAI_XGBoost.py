import xgboost as xgb
import matplotlib.pyplot as plt

model_path = "./models/XGBoost/without_recommendations/dataset_1_strategy_1_model.json"
model = xgb.XGBClassifier()
model.load_model(model_path)

xgb.plot_importance(
    model,
    max_num_features=10,
    importance_type="gain",
    title="Feature Importance (Gain)",
)
plt.show()
