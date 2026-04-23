import joblib
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_logistic_regression_coefficients(
    model_path: str, data_path: str, top_n: int = 10
):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    pipeline = joblib.load(model_path)

    lr_model = pipeline.named_steps["lr"]

    coefficients = lr_model.coef_[0]

    df = pl.read_parquet(data_path)
    feature_names = df.drop("is_hit").columns

    if len(feature_names) != len(coefficients):
        print(
            "Error: Number of features in data does not match the model's coefficients."
        )
        return

    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})

    coef_df = coef_df.sort_values(by="coefficient", ascending=False)

    top_positive = coef_df.head(top_n)
    top_negative = coef_df.tail(top_n)

    plot_data = pd.concat([top_positive, top_negative]).sort_values(
        by="coefficient", ascending=True
    )

    plt.figure(figsize=(12, 8))

    colors = ["#ff6b6b" if x < 0 else "#4ecdc4" for x in plot_data["coefficient"]]

    bars = plt.barh(plot_data["feature"], plot_data["coefficient"], color=colors)

    plt.axvline(x=0, color="black", linestyle="-", linewidth=1.5, alpha=0.5)

    plt.title(
        f"Top {top_n} Positive and Negative Features for Success (Logistic Regression)",
        fontsize=14,
        pad=20,
    )
    plt.xlabel("Standardized Coefficient Value (Impact on Log-Odds)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.05 if width > 0 else width - 0.05
        ha = "left" if width > 0 else "right"
        plt.text(
            label_x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    target_model_path = "./models/Regression/dataset_1_LR_strategy_2_model.joblib"
    target_data_path = "./archive/gamesV1.parquet"

    print("Analizing Feature Importance...")
    plot_logistic_regression_coefficients(target_model_path, target_data_path, top_n=10)
