import pandas as pd

from config import (
    IEEE_TRANSACTION_PATH,
    IEEE_IDENTITY_PATH,
    TARGET_COL,
    TIME_COL_IEEE,
    TRAIN_RATIO,
    VALID_RATIO,
    OUTPUT_DIR,
)

from src.data_loader import load_ieee
from src.splitting import time_train_valid_test_split
from src.preprocessing import prepare_basic_features
from src.models import (
    train_logistic_regression,
    train_random_forest,
    train_lightgbm,
    train_xgboost,
    train_catboost,
    train_isolation_forest,
    get_model_proba,
    get_isolation_score,
)

from src.stacking import (
    build_meta_features,
    train_logistic_meta_model,
    predict_meta_model,
    get_meta_model_coefficients,
)

from src.calibration import (
    fit_platt_scaler,
    apply_platt_scaler,
    fit_isotonic_scaler,
    apply_isotonic_scaler,
    calibration_report,
    get_calibration_curve_dataframe,
)

from src.explain import (
    save_lightgbm_feature_importance,
    save_shap_summary_plot,
    save_shap_bar_plot,
    save_local_shap_waterfall,
)
from src.metrics import evaluate_binary_classifier, results_to_dataframe
from src.threshold_opt import find_best_threshold, cost_based_threshold
from sklearn.metrics import average_precision_score
from src.optimization import optimize_lightgbm
from src.features import prepare_feature_engineering
from src.drift import evaluate_temporal_slices, psi_by_test_slice

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    print("Loading IEEE-CIS dataset...")
    df = load_ieee(IEEE_TRANSACTION_PATH, IEEE_IDENTITY_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Fraud rate:")
    print(df[TARGET_COL].value_counts(normalize=True))

    print("Creating time-based train/valid/test split...")
    train_df, valid_df, test_df = time_train_valid_test_split(
        df,
        time_col=TIME_COL_IEEE,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
    )

    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape}")
    print(f"Test: {test_df.shape}")
    
    print("Applying feature engineering...")
    train_df, valid_df, test_df, feature_info = prepare_feature_engineering(
        train_df,
        valid_df,
        test_df,
    )
    
    print(f"Train after feature engineering: {train_df.shape}")
    print(f"Valid after feature engineering: {valid_df.shape}")
    print(f"Test after feature engineering: {test_df.shape}")

    print("Preprocessing...")
    X_train, y_train, X_valid, y_valid, X_test, y_test, prep_info = prepare_basic_features(
        train_df,
        valid_df,
        test_df,
        target_col=TARGET_COL,
    )

    print(f"Prepared X_train: {X_train.shape}")
    print(f"Prepared X_valid: {X_valid.shape}")
    print(f"Prepared X_test: {X_test.shape}")
    print(f"Dropped columns: {len(prep_info['cols_to_drop'])}")

    models = {}

    print("Training Logistic Regression...")
    models["Logistic Regression"] = train_logistic_regression(X_train, y_train)

    print("Training Random Forest...")
    models["Random Forest"] = train_random_forest(X_train, y_train)

    print("Training LightGBM...")
    models["LightGBM"] = train_lightgbm(X_train, y_train)

    RUN_OPTUNA = False

    if RUN_OPTUNA:
        print("Optimizing LightGBM with Optuna...")
        tuned_lgbm, lgbm_study = optimize_lightgbm(
            X_train,
            y_train,
            X_valid,
            y_valid,
            n_trials=10,
            random_state=42,
        )

        tuned_proba_check = tuned_lgbm.predict_proba(X_valid)[:, 1]
        tuned_pr_auc_check = average_precision_score(y_valid, tuned_proba_check)

        baseline_lgbm_proba_check = models["LightGBM"].predict_proba(X_valid)[:, 1]
        baseline_lgbm_pr_auc_check = average_precision_score(y_valid, baseline_lgbm_proba_check)

        print("Best LightGBM PR-AUC from Optuna study:", lgbm_study.best_value)
        print("Best LightGBM params:", lgbm_study.best_params)
        print("Baseline LightGBM PR-AUC:", baseline_lgbm_pr_auc_check)
        print("Tuned LightGBM PR-AUC:", tuned_pr_auc_check)

        if tuned_pr_auc_check > baseline_lgbm_pr_auc_check:
            print("Tuned LightGBM improved PR-AUC. Adding it to model list.")
            models["LightGBM Tuned"] = tuned_lgbm
        else:
            print("Tuned LightGBM did not improve PR-AUC. Keeping baseline LightGBM only.")

    print("Training XGBoost...")
    models["XGBoost"] = train_xgboost(X_train, y_train)

    print("Training CatBoost...")
    models["CatBoost"] = train_catboost(X_train, y_train)

    print("Training Isolation Forest...")
    iso_model = train_isolation_forest(X_train, y_train)

    results_valid = {}
    predictions_valid = pd.DataFrame({"y_true": y_valid.values})
    predictions_test = pd.DataFrame({"y_true": y_test.values})

    print("Evaluating on validation set...")
    for name, model in models.items():
        valid_proba = get_model_proba(model, X_valid)
        test_proba = get_model_proba(model, X_test)

        predictions_valid[name] = valid_proba
        predictions_test[name] = test_proba

        results_valid[name] = evaluate_binary_classifier(y_valid, valid_proba, threshold=0.5)
        
    
    iso_score_valid = get_isolation_score(iso_model, X_valid)
    iso_score_test = get_isolation_score(iso_model, X_test)

    predictions_valid["Isolation Forest"] = iso_score_valid
    predictions_test["Isolation Forest"] = iso_score_test

    results_valid["Isolation Forest"] = evaluate_binary_classifier(
        y_valid, iso_score_valid, threshold=0.5
    )

    print("Training validation-based stacking meta-model...")

    X_meta_valid, y_meta_valid = build_meta_features(predictions_valid, target_col="y_true")

    meta_model = train_logistic_meta_model(X_meta_valid, y_meta_valid)

    stack_valid_proba = predict_meta_model(
        meta_model,
        predictions_valid,
        target_col="y_true",
    )

    stack_test_proba = predict_meta_model(
        meta_model,
        predictions_test,
        target_col="y_true",
    )

    predictions_valid["Stacking Logistic Meta"] = stack_valid_proba
    predictions_test["Stacking Logistic Meta"] = stack_test_proba

    results_valid["Stacking Logistic Meta"] = evaluate_binary_classifier(
        y_valid,
        stack_valid_proba,
        threshold=0.5,
    )

    coef_df = get_meta_model_coefficients(
        meta_model,
        X_meta_valid.columns.tolist(),
    )

    print("\nStacking meta-model coefficients:")
    print(coef_df.to_string(index=False))

    coef_df.to_csv(
        OUTPUT_DIR / "reports" / "stacking_meta_model_coefficients.csv",
        index=False,
    )

    results_df = results_to_dataframe(results_valid)

    print("\nValidation results:")
    print(results_df)

    results_df.to_csv(OUTPUT_DIR / "reports" / "validation_baseline_results.csv")
    predictions_valid.to_csv(
        OUTPUT_DIR / "predictions" / "validation_predictions.csv",
        index=False,
    )

    predictions_test.to_csv(
        OUTPUT_DIR / "predictions" / "test_predictions.csv",
        index=False,
    )
    
    print("\nEvaluating saved model predictions on test set at threshold 0.5...")

    results_test = {}

    for model_name in predictions_test.columns:
        if model_name == "y_true":
            continue

        y_proba_test = predictions_test[model_name].values

        results_test[model_name] = evaluate_binary_classifier(
            y_test,
            y_proba_test,
            threshold=0.5,
        )

    results_test_df = results_to_dataframe(results_test)

    print("\nTest results at threshold 0.5:")
    print(results_test_df)

    results_test_df.to_csv(
        OUTPUT_DIR / "reports" / "test_results_threshold_05.csv"
    )
    
    print("\nThreshold optimization on validation set...")

    threshold_reports = []

    valid_amounts = valid_df["TransactionAmt"].values if "TransactionAmt" in valid_df.columns else None

    for model_name in predictions_valid.columns:
        if model_name == "y_true":
            continue

        y_proba = predictions_valid[model_name].values

        best_f2, threshold_df = find_best_threshold(y_valid, y_proba, metric="f2")

        threshold_df.to_csv(
            OUTPUT_DIR / "reports" / f"thresholds_{model_name.replace(' ', '_')}.csv",
            index=False
        )

        best_cost, best_benefit, cost_df = cost_based_threshold(
            y_true=y_valid,
            y_proba=y_proba,
            amounts=valid_amounts,
            fp_cost=1.0,
            fn_cost=100.0,
        )

        cost_df.to_csv(
            OUTPUT_DIR / "reports" / f"cost_thresholds_{model_name.replace(' ', '_')}.csv",
            index=False
        )

        threshold_reports.append({
            "model": model_name,
            "best_f2_threshold": best_f2["threshold"],
            "best_f2": best_f2["f2"],
            "precision_at_best_f2": best_f2["precision"],
            "recall_at_best_f2": best_f2["recall"],
            "mcc_at_best_f2": best_f2["mcc"],
            "best_cost_threshold": best_cost["threshold"],
            "min_total_cost": best_cost["total_cost"],
            "best_benefit_threshold": best_benefit["threshold"],
            "max_net_benefit": best_benefit["net_benefit"],
        })

    threshold_summary = pd.DataFrame(threshold_reports)

    print("\nThreshold optimization summary:")
    print(threshold_summary.sort_values("best_f2", ascending=False))

    threshold_summary.to_csv(
        OUTPUT_DIR / "reports" / "threshold_optimization_summary.csv",
        index=False
    )
    
    print("\nEvaluating test set using validation-selected F2 thresholds...")

    test_threshold_results = {}

    for _, row in threshold_summary.iterrows():
        model_name = row["model"]

        if model_name not in predictions_test.columns:
            continue

        threshold = row["best_f2_threshold"]
        y_proba_test = predictions_test[model_name].values

        test_threshold_results[model_name] = evaluate_binary_classifier(
            y_test,
            y_proba_test,
            threshold=threshold,
        )

        test_threshold_results[model_name]["validation_selected_threshold"] = threshold

    test_threshold_df = results_to_dataframe(test_threshold_results)

    print("\nTest results using validation-selected F2 thresholds:")
    print(test_threshold_df)

    test_threshold_df.to_csv(
        OUTPUT_DIR / "reports" / "test_results_validation_f2_thresholds.csv"
    )

    print("\nCalibration analysis for LightGBM and Stacking Logistic Meta...")

    calibration_rows = []
    calibration_curves = []

    models_to_calibrate = ["LightGBM", "Stacking Logistic Meta"]

    for model_name in models_to_calibrate:
        if model_name not in predictions_valid.columns or model_name not in predictions_test.columns:
            continue

        valid_proba = predictions_valid[model_name].values
        test_proba = predictions_test[model_name].values

        # Uncalibrated reports
        calibration_rows.append(
            calibration_report(y_test, test_proba, f"{model_name} uncalibrated")
        )

        calibration_curves.append(
            get_calibration_curve_dataframe(
                y_test,
                test_proba,
                f"{model_name} uncalibrated",
                n_bins=10,
            )
        )

        # Platt scaling fitted on validation, applied to test
        platt = fit_platt_scaler(y_valid, valid_proba)
        test_platt = apply_platt_scaler(platt, test_proba)

        predictions_test[f"{model_name} Platt Calibrated"] = test_platt

        calibration_rows.append(
            calibration_report(y_test, test_platt, f"{model_name} Platt")
        )

        calibration_curves.append(
            get_calibration_curve_dataframe(
                y_test,
                test_platt,
                f"{model_name} Platt",
                n_bins=10,
            )
        )

        # Isotonic fitted on validation, applied to test
        isotonic = fit_isotonic_scaler(y_valid, valid_proba)
        test_iso = apply_isotonic_scaler(isotonic, test_proba)

        predictions_test[f"{model_name} Isotonic Calibrated"] = test_iso

        calibration_rows.append(
            calibration_report(y_test, test_iso, f"{model_name} Isotonic")
        )

        calibration_curves.append(
            get_calibration_curve_dataframe(
                y_test,
                test_iso,
                f"{model_name} Isotonic",
                n_bins=10,
            )
        )

    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df = pd.DataFrame(calibration_rows)

    if calibration_curves:
        calibration_curve_df = pd.concat(calibration_curves, ignore_index=True)
    else:
        calibration_curve_df = pd.DataFrame()

    print("\nCalibration report on test set:")
    print(calibration_df.to_string(index=False))

    calibration_df.to_csv(
        OUTPUT_DIR / "reports" / "calibration_report_test.csv",
        index=False,
    )

    calibration_curve_df.to_csv(
        OUTPUT_DIR / "reports" / "calibration_curves_test.csv",
        index=False,
    )

    predictions_test.to_csv(
        OUTPUT_DIR / "predictions" / "test_predictions_with_calibration.csv",
        index=False,
    )
    
    print("\nDrift analysis on test set...")

    drift_model_names = [
        "LightGBM",
        "Stacking Logistic Meta",
        "XGBoost",
        "CatBoost",
    ]

    temporal_results = evaluate_temporal_slices(
        test_df=test_df,
        predictions_test=predictions_test,
        y_test=y_test,
        time_col=TIME_COL_IEEE,
        model_names=drift_model_names,
        threshold=0.5,
        n_slices=4,
    )

    print("\nTemporal slice performance:")
    print(temporal_results.to_string(index=False))

    temporal_results.to_csv(
        OUTPUT_DIR / "reports" / "temporal_slice_performance_test.csv",
        index=False,
    )

    psi_results = psi_by_test_slice(
        predictions_valid=predictions_valid,
        predictions_test=predictions_test,
        test_df=test_df,
        time_col=TIME_COL_IEEE,
        model_names=["LightGBM", "Stacking Logistic Meta"],
        n_slices=4,
    )

    print("\nPSI results:")
    print(psi_results.to_string(index=False))

    psi_results.to_csv(
        OUTPUT_DIR / "reports" / "psi_predictions_test_slices.csv",
        index=False,
    )
    
    print("\nGenerating SHAP explanations for LightGBM...")

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Use a sample to keep SHAP fast
    shap_sample_size = min(3000, len(X_test))
    X_shap_sample = X_test.sample(
        n=shap_sample_size,
        random_state=42,
    )

    top_importance = save_lightgbm_feature_importance(
        model=models["LightGBM"],
        X_train=X_train,
        output_path=OUTPUT_DIR / "reports" / "lightgbm_feature_importance.csv",
        top_n=30,
    )

    print("\nTop 30 LightGBM feature importances:")
    print(top_importance.to_string(index=False))

    save_shap_summary_plot(
        model=models["LightGBM"],
        X_sample=X_shap_sample,
        output_path=figures_dir / "lightgbm_shap_summary.png",
    )

    save_shap_bar_plot(
        model=models["LightGBM"],
        X_sample=X_shap_sample,
        output_path=figures_dir / "lightgbm_shap_bar.png",
    )

    # Pick one fraud example from test sample if available
    fraud_positions = [i for i, value in enumerate(y_test.values) if value == 1]

    if fraud_positions:
        row_position = fraud_positions[0]
        X_local = X_test.iloc[[row_position]]

        save_local_shap_waterfall(
            model=models["LightGBM"],
            X_sample=X_local,
            row_index=0,
            output_path=figures_dir / "lightgbm_local_fraud_waterfall.png",
        )

        print("Saved local SHAP waterfall for one fraud transaction.")

    print("\nSaved results to outputs/reports and outputs/predictions.")


if __name__ == "__main__":
    main()