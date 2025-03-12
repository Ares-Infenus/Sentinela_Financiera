"""Este módulo define la clase XGBOptunaTuner, que se encarga de optimizar y evaluar
un clasificador XGBoost utilizando Optuna. Entre sus funcionalidades se encuentran:

Optimización de hiperparámetros: Mediante validación cruzada se ajustan parámetros como el
número de estimadores, profundidad máxima, tasa de aprendizaje, entre otros, penalizando
configuraciones complejas.
Entrenamiento del mejor modelo: Una vez identificados los mejores parámetros, se entrena el
modelo con dichos ajustes.
Evaluación del modelo: Se calculan métricas de rendimiento en conjuntos de prueba y validación.
Guardado de resultados: Los resultados se exportan a un archivo CSV para su posterior análisis.
El módulo incluye también un ejemplo de uso utilizando el conjunto de datos de cáncer de mama
de scikit-learn.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import optuna


# Constantes
RANDOM_STATE = 42
N_SPLITS = 5
PENALTY_COEF = 0.001


class XGBOptunaTuner:
    """
    Optimiza un clasificador XGBoost usando Optuna.
    """

    def __init__(self, train_x, train_y, test_x, test_y, valid_x=None, valid_y=None):
        self.train_x, self.train_y = train_x, train_y
        self.test_x, self.test_y = test_x, test_y
        self.valid_x, self.valid_y = valid_x, valid_y
        self.study = None
        self.best_model = None
        self.results_df = None

    def _objective(self, trial):
        """
        Función objetivo para evaluar hiperparámetros.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "use_label_encoder": False,
        }
        model = xgb.XGBClassifier(**params)
        cv_score = np.mean(
            cross_val_score(
                model,
                self.train_x,
                self.train_y,
                cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
                scoring="accuracy",
            )
        )
        penalty = (
            PENALTY_COEF
            * params["n_estimators"]
            * params["max_depth"]
            * params["colsample_bytree"]
        )
        return cv_score - penalty

    def optimize(self, n_trials=50, storage_path=None):
        """
        Ejecuta la optimización de hiperparámetros.
        """
        storage = f"sqlite:///{storage_path}" if storage_path else None
        self.study = optuna.create_study(
            direction="maximize", storage=storage, load_if_exists=bool(storage)
        )
        self.study.optimize(self._objective, n_trials=n_trials)
        print(
            f"Mejor trial: {self.study.best_trial.value:.4f}, Parámetros: {self.study.best_trial.params}"  # pylint: disable=line-too-long
        )

    def train_best_model(self):
        """
        Entrena el mejor modelo con los parámetros óptimos.
        """
        self.best_model = xgb.XGBClassifier(
            **self.study.best_trial.params,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            use_label_encoder=False,
        )
        self.best_model.fit(self.train_x, self.train_y)

    def evaluate(self):
        """
        Evalúa el modelo en datos de prueba y validación.
        """
        test_acc = accuracy_score(self.test_y, self.best_model.predict(self.test_x))
        valid_acc = (
            accuracy_score(self.valid_y, self.best_model.predict(self.valid_x))
            if self.valid_x is not None
            else None
        )
        self.results_df = pd.DataFrame(
            {
                "Métrica": ["Objective Score", "Test Accuracy", "Validation Accuracy"],
                "Valor": [self.study.best_trial.value, test_acc, valid_acc],
            }
        )
        print(self.results_df)

    def save_results(self, path):
        """
        Guarda los resultados en un archivo CSV.
        """
        self.results_df.to_csv(path, index=False)
        print(f"Resultados guardados en: {path}")


if __name__ == "__main__":
    data = load_breast_cancer()
    X_train, X_temp, y_train, y_temp = train_test_split(
        data.data,  # pylint: disable=no-member
        data.target,  # pylint: disable=no-member
        test_size=0.4,
        random_state=42,  # pylint: disable=no-member
    )
    X_test, X_valid, y_test, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    tuner = XGBOptunaTuner(X_train, y_train, X_test, y_test, X_valid, y_valid)
    tuner.optimize(n_trials=50, storage_path="optuna_trialsw.db")
    tuner.train_best_model()
    tuner.evaluate()
    tuner.save_results("resultados_finales.csv")
