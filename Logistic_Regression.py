import os
import polars as pl
import numpy as np
import mlflow
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def get_split_50_50(df: pl.DataFrame, target_col: str, seed: int):
    df_indexed = df.with_row_index("idx")
    y_pandas = df_indexed.select("idx", target_col).to_pandas()

    train_idx, test_idx = train_test_split(
        y_pandas["idx"],
        test_size=0.50,
        stratify=y_pandas[target_col],
        random_state=seed,
    )

    df_train = df_indexed.filter(pl.col("idx").is_in(train_idx.tolist())).drop("idx")
    df_test = df_indexed.filter(pl.col("idx").is_in(test_idx.tolist())).drop("idx")

    return df_train, df_test


def get_split_9_91(df: pl.DataFrame, target_col: str, seed: int):
    df_indexed = df.with_row_index("idx")

    total_rows = df_indexed.height
    target_train_hits = int(0.03 * total_rows)
    target_train_non_hits = int(0.06 * total_rows)

    hits_pd = df_indexed.filter(pl.col(target_col) == 1).select("idx").to_pandas()
    non_hits_pd = df_indexed.filter(pl.col(target_col) == 0).select("idx").to_pandas()

    target_train_hits = min(target_train_hits, len(hits_pd) - 1)
    target_train_non_hits = min(target_train_non_hits, len(non_hits_pd) - 1)

    train_hits_idx, test_hits_idx = train_test_split(
        hits_pd["idx"], train_size=target_train_hits, random_state=seed
    )

    train_non_hits_idx, test_non_hits_idx = train_test_split(
        non_hits_pd["idx"], train_size=target_train_non_hits, random_state=seed
    )

    train_idx_combined = train_hits_idx.tolist() + train_non_hits_idx.tolist()
    test_idx_combined = test_hits_idx.tolist() + test_non_hits_idx.tolist()

    df_train = (
        df_indexed.filter(pl.col("idx").is_in(train_idx_combined))
        .drop("idx")
        .sample(fraction=1.0, seed=seed)
    )
    df_test = (
        df_indexed.filter(pl.col("idx").is_in(test_idx_combined))
        .drop("idx")
        .sample(fraction=1.0, seed=seed)
    )

    return df_train, df_test


def main():
    target_column = "is_hit"
    global_seed = 123
    cv_folds = 5
    models_dir = "./models/Regression"
    os.makedirs(models_dir, exist_ok=True)

    datasets_config = {
        "dataset_1": "./archive/gamesV1.parquet",
        "dataset_2": "./archive/gamesV2.parquet",
    }

    mlflow.set_experiment("Novomatic_LogisticRegression_Evaluation")

    for dataset_name, dataset_path in datasets_config.items():
        print(f"\n--- Processing dataset: {dataset_name} ---")

        if not os.path.exists(dataset_path):
            print(f"File {dataset_path} does not exist. Skipping.")
            continue

        df_main = pl.read_parquet(dataset_path)

        strategies = [4, 3, 2, 1]

        for strategy_id in strategies:
            print(f"\nTraining {dataset_name} - Strategy {strategy_id}")
            run_name = f"{dataset_name}_LR_strategy_{strategy_id}"

            with mlflow.start_run(run_name=run_name):
                use_undersampling = strategy_id in [3, 4]
                use_class_weights = strategy_id in [2, 4]

                if use_undersampling:
                    df_train, df_test = get_split_9_91(
                        df_main, target_column, global_seed
                    )
                else:
                    df_train, df_test = get_split_50_50(
                        df_main, target_column, global_seed
                    )

                X_train_full = df_train.drop(target_column).to_pandas()
                y_train_full = df_train.select(target_column).to_numpy().ravel()

                X_test = df_test.drop(target_column).to_pandas()
                y_test = df_test.select(target_column).to_numpy().ravel()

                mlflow.log_params(
                    {
                        "model_type": "LogisticRegression",
                        "strategy_id": strategy_id,
                        "use_undersampling": use_undersampling,
                        "use_class_weights": use_class_weights,
                        "random_state": global_seed,
                        "cv_folds": cv_folds,
                        "train_size_rows": X_train_full.shape[0],
                        "test_size_rows": X_test.shape[0],
                        "scaled": True,
                    }
                )

                skf = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=global_seed
                )

                val_metrics_list = []
                best_model = None
                best_val_f1 = -1.0

                for fold_idx, (train_idx, val_idx) in enumerate(
                    skf.split(X_train_full, y_train_full)
                ):
                    X_fold_train, X_fold_val = (
                        X_train_full.iloc[train_idx],
                        X_train_full.iloc[val_idx],
                    )
                    y_fold_train, y_fold_val = (
                        y_train_full[train_idx],
                        y_train_full[val_idx],
                    )

                    class_weight_param = "balanced" if use_class_weights else None

                    pipeline = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "lr",
                                LogisticRegression(
                                    class_weight=class_weight_param,
                                    random_state=global_seed,
                                    max_iter=2000,
                                ),
                            ),
                        ]
                    )

                    pipeline.fit(X_fold_train, y_fold_train)

                    val_preds = pipeline.predict(X_fold_val)
                    current_val_f1 = f1_score(y_fold_val, val_preds, zero_division=0)

                    val_metrics_list.append(
                        {
                            "val_precision": precision_score(
                                y_fold_val, val_preds, zero_division=0
                            ),
                            "val_recall": recall_score(
                                y_fold_val, val_preds, zero_division=0
                            ),
                            "val_f1": current_val_f1,
                            "val_accuracy": accuracy_score(y_fold_val, val_preds),
                        }
                    )

                    if current_val_f1 > best_val_f1:
                        best_val_f1 = current_val_f1
                        best_model = pipeline

                avg_val_metrics = {
                    k: np.mean([fold[k] for fold in val_metrics_list])
                    for k in val_metrics_list[0].keys()
                }
                mlflow.log_metrics(avg_val_metrics)

                print(
                    f"Strategy {strategy_id} CV Validation Results (Averaged over 5 folds):"
                )
                print(
                    f"Val Precision: {avg_val_metrics['val_precision']:.4f} | Val Recall: {avg_val_metrics['val_recall']:.4f} | Val F1: {avg_val_metrics['val_f1']:.4f}"
                )

                if best_model is not None:
                    test_preds = best_model.predict(X_test)

                    test_metrics = {
                        "test_precision": precision_score(
                            y_test, test_preds, zero_division=0
                        ),
                        "test_recall": recall_score(
                            y_test, test_preds, zero_division=0
                        ),
                        "test_f1": f1_score(y_test, test_preds, zero_division=0),
                        "test_accuracy": accuracy_score(y_test, test_preds),
                    }

                    mlflow.log_metrics(test_metrics)

                    print("\nFINAL TEST Results (on unseen 50% data):")
                    print(
                        f"Test Precision: {test_metrics['test_precision']:.4f} | Test Recall: {test_metrics['test_recall']:.4f} | Test F1: {test_metrics['test_f1']:.4f}"
                    )

                    model_save_path = os.path.join(
                        models_dir,
                        f"{dataset_name}_LR_strategy_{strategy_id}_model.joblib",
                    )
                    joblib.dump(best_model, model_save_path)
                    print(f"Best pipeline saved to: {model_save_path}\n")


if __name__ == "__main__":
    main()
