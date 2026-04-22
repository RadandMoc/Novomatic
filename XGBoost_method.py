import os
import polars as pl
import numpy as np
import xgboost as xgb
import mlflow
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# strategie eksperymentu:
# 1. Bez ingerencji w dysbalans danych (bez ważenia klas i undersamplingu) (50% zbioru treningowego)
# 2. Z ważeniem klas hiperparametrami (50% zbioru treningowego)
# 3. Z undersamplingiem (ale bez ważenia klas) (9% danych na zbiór treningowy, czyli 6% niehitów i 3% hitów)
# 4. Z undersamplingiem i ważeniem klas hiperparametrami (tak jak wyżej 9% na zbiór treningowy)


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
    models_dir = "./models/XGBoost"
    os.makedirs(models_dir, exist_ok=True)

    datasets_config = {
        "dataset_1": "./archive/gamesV1.parquet",
        "dataset_2": "./archive/gamesV2.parquet",
    }

    mlflow.set_experiment("Novomatic_XGBoost_Evaluation")

    for dataset_name, dataset_path in datasets_config.items():
        print(f"\n--- Processing dataset: {dataset_name} ---")

        if not os.path.exists(dataset_path):
            print(f"File {dataset_path} does not exist. Skipping.")
            continue

        df_main = pl.read_parquet(dataset_path)

        strategies = [4, 3, 2, 1]

        for strategy_id in strategies:
            print(f"\nTraining {dataset_name} - Strategy {strategy_id}")
            run_name = f"{dataset_name}_strategy_{strategy_id}"

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
                        "strategy_id": strategy_id,
                        "use_undersampling": use_undersampling,
                        "use_class_weights": use_class_weights,
                        "random_state": global_seed,
                        "cv_folds": cv_folds,
                        "train_size_rows": X_train_full.shape[0],
                        "test_size_rows": X_test.shape[0],
                    }
                )

                skf = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=global_seed
                )

                cv_metrics_list = []
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

                    if use_class_weights:
                        n_pos = np.sum(y_fold_train == 1)
                        n_neg = np.sum(y_fold_train == 0)
                        spw = float(n_neg / n_pos) if n_pos > 0 else 1.0
                    else:
                        spw = 1.0

                    model = xgb.XGBClassifier(
                        n_estimators=500,
                        scale_pos_weight=spw,
                        random_state=global_seed,
                        eval_metric="logloss",
                        early_stopping_rounds=20,
                    )

                    model.fit(
                        X_fold_train,
                        y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        verbose=False,
                    )

                    val_preds = model.predict(X_fold_val)
                    current_val_f1 = f1_score(y_fold_val, val_preds, zero_division=0)

                    if current_val_f1 > best_val_f1:
                        best_val_f1 = current_val_f1
                        best_model = model

                    test_preds = model.predict(X_test)
                    fold_metrics = {
                        "test_precision": precision_score(
                            y_test, test_preds, zero_division=0
                        ),
                        "test_recall": recall_score(
                            y_test, test_preds, zero_division=0
                        ),
                        "test_f1": f1_score(y_test, test_preds, zero_division=0),
                        "test_accuracy": accuracy_score(y_test, test_preds),
                    }
                    cv_metrics_list.append(fold_metrics)

                avg_metrics = {}
                for key in cv_metrics_list[0].keys():
                    avg_metrics[key] = np.mean([fold[key] for fold in cv_metrics_list])

                mlflow.log_metrics(avg_metrics)
                mlflow.log_metric("best_cv_val_f1", best_val_f1)

                print(
                    f"Strategy {strategy_id} CV Results (Averaged over {cv_folds} folds):"
                )
                print(
                    f"Precision: {avg_metrics['test_precision']:.4f} | Recall: {avg_metrics['test_recall']:.4f} | F1: {avg_metrics['test_f1']:.4f}"
                )

                if best_model is not None:
                    model_save_path = os.path.join(
                        models_dir, f"{dataset_name}_strategy_{strategy_id}_model.json"
                    )
                    best_model.save_model(model_save_path)
                    print(f"Best model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
