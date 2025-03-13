"""Centralizes all system configuration in a CONFIG dictionary. Stores file paths,
model parameters, training configurations and data partitioning. Acts as a single
source of truth for configurable parameters, facilitating global changes without
the need to modify the code."""

CONFIG = {
    "data": {"raw_path": "data/raw/fraud_data.csv", "processed_dir": "data/processed/"},
    "models": {
        "enabled": ["neural_network", "xgboost", "logistic_regression"],
        "neural_network": {
            "n_trials": 60,
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
        "xgboost": {
            "n_trials": 50,
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
        "logistic_regression": {
            "script_path": "fraud_models/logistic_fraud_model.r",
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
    },
    "output": {"directory": "results/model_performance/", "logs_dir": "logs/"},
    "split_ratio": "7-1.5-1.5",  # train-validation-test ratio
    "target_column": "isFraud",
    "exclude_columns": ["id","Amount","Class"],
    "exclude_direction": r"C:\Users\spinz\OneDrive\Documentos\Portafolio oficial\Sentinela_financiera\data\processed\Datosiqr\filtered_data.csv",
    "train_test_val_direction": r"C:\Users\spinz\OneDrive\Documentos\Portafolio oficial\Sentinela_financiera\data\processed\train_test_validation",
    "dataset_export_folder": r"C:\Users\spinz\OneDrive\Documentos\Portafolio oficial\Sentinela_financiera\data\processed\train_test_validation_scaled",
}
