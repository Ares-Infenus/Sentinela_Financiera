"""Implements the Factory design pattern along with inheritance for the different models. Contains:
Model abstract class that defines the common interface.
Concrete implementations for neural networks, XGBoost and logistic regression.
ModelFactory class that creates instances of the appropriate model according to the configuration.
"""

from abc import ABC, abstractmethod
import os
import subprocess
from typing import Dict, Any

import pandas as pd

from config.settings import CONFIG
from data_preparation.pipeline import DataSet
from fraud_models.neural_networks import NeuralNetworkOptimizer
from fraud_models.xgboost_fraud import XGBOptunaTuner


class Model(ABC):
    """Abstract base class for all models."""

    def __init__(self, train_data: DataSet, val_data: DataSet, test_data: DataSet):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    @abstractmethod
    def train_and_evaluate(self) -> pd.DataFrame:
        """Train the model and evaluate its performance."""


class NeuralNetworkModel(Model):
    """Neural Network implementation of the Model interface."""

    def train_and_evaluate(self) -> pd.DataFrame:
        """Train and evaluate a neural network model."""
        config = CONFIG["models"]["neural_network"]

        optimizer = NeuralNetworkOptimizer(
            self.train_data.x.values,
            self.val_data.x.values,
            self.train_data.y,
            self.val_data.y,
        )

        _, _ = optimizer.optimize(
            n_trials=config["n_trials"],
            study_name="binary_classification_study",
            storage=r"sqlite:///C:\Users\spinz\OneDrive\Documentos\Portafolio oficial\Sentinela_financiera\reports\results\database_neuronal_Network\neuronal_network_optimizacion.db")
        return optimizer.get_results_dataframe()


class XGBoostModel(Model):
    """XGBoost implementation of the Model interface."""

    def train_and_evaluate(self) -> pd.DataFrame:
        """Train and evaluate an XGBoost model."""
        config = CONFIG["models"]["xgboost"]

        tuner = XGBOptunaTuner(
            train_x=self.train_data.x,
            train_y=self.train_data.y,
            test_x=self.test_data.x,
            test_y=self.test_data.y,
            valid_x=self.val_data.x,
            valid_y=self.val_data.y,
        )

        tuner.optimize(n_trials=config["n_trials"])
        return tuner.save_results("resultados_finales.csv")


class LogisticRegressionModel(Model):
    """Logistic Regression implementation using R script."""

    def train_and_evaluate(self) -> pd.DataFrame:
        """Run R script for logistic regression and return results."""
        # Save data for R script
        self._save_data_for_r()

        # Execute R script
        script_path = CONFIG["models"]["logistic_regression"]["script_path"]
        _ = self._run_r_script(script_path)

        # Parse results from R (assuming R script saves results to CSV)
        results_path = os.path.join(
            CONFIG["output"]["directory"], "logistic_regression_results.csv"
        )
        if os.path.exists(results_path):
            return pd.read_csv(results_path)
        else:
            return pd.DataFrame({"error": ["R script failed to save results"]})

    def _save_data_for_r(self):
        """Save data in format expected by R script."""
        output_folder = CONFIG["data"]["processed_dir"]
        os.makedirs(output_folder, exist_ok=True)

        # Save train data
        self.train_data.X.to_csv(
            os.path.join(output_folder, "X_train_scaled.csv"), index=False
        )
        pd.DataFrame(self.train_data.y).to_csv(
            os.path.join(output_folder, "y_train.csv"), index=False
        )

        # Save validation data
        self.val_data.X.to_csv(
            os.path.join(output_folder, "X_val_scaled.csv"), index=False
        )
        pd.DataFrame(self.val_data.y).to_csv(
            os.path.join(output_folder, "y_val.csv"), index=False
        )

        # Save test data
        self.test_data.X.to_csv(
            os.path.join(output_folder, "X_test_scaled.csv"), index=False
        )
        pd.DataFrame(self.test_data.y).to_csv(
            os.path.join(output_folder, "y_test.csv"), index=False
        )

    def _run_r_script(self, script_path: str) -> Dict[str, Any]:
        """Execute R script and return result."""
        try:
            result = subprocess.run(
                ["Rscript", script_path], capture_output=True, text=True, check=True
            )
            return {"success": True, "output": result.stdout}
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": e.stderr}


class ModelFactory:
    """Factory for creating model instances."""

    def __init__(self, train_data: DataSet, val_data: DataSet, test_data: DataSet):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def create_model(self, model_type: str) -> Model:
        """Create and return a model instance based on the type."""
        if model_type == "neural_network":
            return NeuralNetworkModel(self.train_data, self.val_data, self.test_data)
        elif model_type == "xgboost":
            return XGBoostModel(self.train_data, self.val_data, self.test_data)
        elif model_type == "logistic_regression":
            return LogisticRegressionModel(
                self.train_data, self.val_data, self.test_data
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
