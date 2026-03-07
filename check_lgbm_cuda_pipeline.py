from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from skopt import BayesSearchCV

from cuda_training_support import (
    DEFAULT_TARGET_COLUMN,
    build_lgbm_classifier,
    build_notebook_run_config,
    get_lgbm_search_spaces,
    load_training_dataframe,
    prepare_lightgbm_training_data,
    resolve_data_path,
    validate_lightgbm_cuda_build,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the CUDA-only LightGBM training pipeline.")
    parser.add_argument("--run-mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--bayes-n-iter", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_notebook_run_config(
        run_mode=args.run_mode,
        sample_size=args.sample_size,
        bayes_n_iter=args.bayes_n_iter,
        cv_folds=args.cv_folds,
        random_seed=args.random_seed,
    )

    data_path = resolve_data_path()
    print(f"data_path={data_path}")
    print(
        "config="
        f"run_mode={config.run_mode},"
        f" sample_size={config.sample_size},"
        f" bayes_n_iter={config.bayes_n_iter},"
        f" cv_folds={config.cv_folds}"
    )

    data = load_training_dataframe(
        data_path=data_path,
        random_seed=config.random_seed,
        run_mode=config.run_mode,
        sample_size=config.sample_size,
        target_column=DEFAULT_TARGET_COLUMN,
    )
    print(f"loaded_rows={len(data)}")
    print(f"label_distribution=\n{data[DEFAULT_TARGET_COLUMN].value_counts()}")

    prepared = prepare_lightgbm_training_data(
        data,
        target_column=DEFAULT_TARGET_COLUMN,
        random_state=config.random_seed,
        test_size=config.test_size,
    )
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    print(f"train_shape={X_train.shape}")
    print(f"test_shape={X_test.shape}")
    print(f"resampled_train_distribution=\n{pd.Series(y_train).value_counts()}")

    lgbm_version = validate_lightgbm_cuda_build(random_state=config.random_seed)
    print(f"lightgbm_version={lgbm_version}")
    print("cuda_preflight=ok")

    estimator = build_lgbm_classifier(
        random_state=config.random_seed,
        model_n_jobs=config.model_n_jobs,
    )
    search_spaces = get_lgbm_search_spaces(config.run_mode)

    bayes_search = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=config.bayes_n_iter,
        cv=config.cv_folds,
        scoring="roc_auc",
        n_jobs=config.search_n_jobs,
        verbose=1,
        random_state=config.random_seed,
    )

    bayes_search.fit(X_train, y_train)
    best_lgb = bayes_search.best_estimator_

    y_test_pred = best_lgb.predict(X_test)
    y_test_proba = best_lgb.predict_proba(X_test)[:, 1]

    metrics = {
        "AUC": roc_auc_score(y_test, y_test_proba),
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "F1": f1_score(y_test, y_test_pred),
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    metrics["Specificity"] = tn / (tn + fp)

    print(f"best_params={bayes_search.best_params_}")
    for metric, value in metrics.items():
        print(f"{metric}={value:.6f}")
    print(f"confusion_matrix=\n{confusion_matrix(y_test, y_test_pred)}")


if __name__ == "__main__":
    main()
