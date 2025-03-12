"""he code implements a neural network optimization system using Optuna and TensorFlow.

ActivationFunctions: provides custom activation functions such as swish and a factory
method to obtain activations.

ModelEvaluator: Calculates a combined score for models based on accuracy, loss and
number of parameters.

ModelBuilder: Builds Keras models using hyperparameters suggested by Optuna.

TrainingConfig: Manages training configurations such as callbacks, optimizers and batch sizes.

DataHandler: Prepares and manages training and validation data.
NeuralNetworkOptimizer: The main class that composes all the previous ones to execute
hyperparameter optimization.

The system allows:

Automatically search for the best configuration for neural networks.
Evaluate models using weighted combined metrics.
Export results for analysis
Build the best model found during optimization"""

import time
from typing import Dict, Tuple, List, Any, Callable, Union, Optional
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential  # pylint: disable = E0401, E0611
from tensorflow.keras.layers import (  # pylint: disable = E0401, E0611
    Dense,
    Dropout,
    LeakyReLU,
)
from tensorflow.keras.callbacks import EarlyStopping  # pylint: disable = E0401, E0611
from tensorflow.keras.optimizers import Adam, RMSprop  # pylint: disable = E0401, E0611
from optuna.integration import TFKerasPruningCallback


# Define activation functions as a separate module for Single Responsibility Principle
# Constantes para nombres de funciones de activación soportadas
LEAKY_RELU = "leaky_relu"
SWISH = "swish"
DEFAULT_LEAKY_ALPHA = 0.3  # Valor por defecto para Leaky ReLU


class ActivationFunctions:
    """
    Proporciona funciones de activación para modelos de redes neuronales.
    Cada método está enfocado a una única tarea para mantener la claridad.
    """

    @staticmethod
    def swish(x: tf.Tensor) -> tf.Tensor:
        """
        Calcula la activación Swish: f(x) = x * sigmoid(x).

        Parámetros:
            x: tf.Tensor de entrada.

        Retorna:
            tf.Tensor resultante de aplicar Swish.
        """
        # Se asume que 'x' es un tensor de TensorFlow
        return x * tf.sigmoid(x)

    @staticmethod
    def get_activation(
        name: str, alpha: Optional[float] = None
    ) -> Callable[[tf.Tensor], tf.Tensor]:
        """
        Función fábrica que retorna la función de activación según el nombre proporcionado.
        Utiliza estructuras de control simples para seleccionar la activación.

        Parámetros:
            name (str): Nombre de la activación ("leaky_relu" o "swish").
            alpha (Optional[float]): Coeficiente para Leaky ReLU. Si no se especifica,
                                       se utiliza DEFAULT_LEAKY_ALPHA.

        Retorna:
            Callable[[tf.Tensor], tf.Tensor]: Función que transforma un tensor de entrada.

        Lanza:
            ValueError: Si se especifica un nombre de activación no soportado.
        """
        # Inicializa 'alpha' con un valor por defecto si es None
        if name == LEAKY_RELU:
            alpha_valor = alpha if alpha is not None else DEFAULT_LEAKY_ALPHA
            # tf.keras.layers.LeakyReLU retorna una capa callable
            return tf.keras.layers.LeakyReLU(alpha=alpha_valor)  # pylint: disable=E1101
        elif name == SWISH:
            return ActivationFunctions.swish
        else:
            raise ValueError(f"Función de activación no soportada: {name}")


# Model evaluation with Single Responsibility Principle
# Constantes para valores máximos y comparaciones de flotantes
DEFAULT_MAX_PRECISION = 1.0
DEFAULT_MAX_LOSS = 1.0
DEFAULT_MAX_PARAMS = int(1e6)
EPSILON = 1e-6  # Tolerancia para validación de suma de pesos


class ModelEvaluator:
    """
    Evalúa modelos de machine learning utilizando métricas de precisión, pérdida
    y cantidad de parámetros. Calcula un puntaje combinado para facilitar la
    comparación y selección de modelos.
    """

    def __init__(
        self,
        max_precision: float = DEFAULT_MAX_PRECISION,
        max_loss: float = DEFAULT_MAX_LOSS,
        max_params: int = DEFAULT_MAX_PARAMS,
    ):
        """
        Inicializa el evaluador con los máximos valores esperados para cada métrica.

        Parámetros:
            max_precision (float): Precisión máxima esperada.
            max_loss (float): Pérdida máxima esperada.
            max_params (int): Cantidad máxima de parámetros esperados.
        """
        self.max_precision = max_precision
        self.max_loss = max_loss
        self.max_params = max_params

    def calculate_score(
        self,
        precision: float,
        loss: float,
        num_params: int,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> float:
        """
        Calcula un puntaje combinado basado en la normalización de las métricas del modelo.

        Se normalizan:
          - La precisión (valor mayor es mejor).
          - La pérdida (valor menor es mejor; se invierte la escala).
          - La cantidad de parámetros (menos es mejor; se invierte la escala).

        Parámetros:
            precision (float): Precisión del modelo.
            loss (float): Pérdida del modelo.
            num_params (int): Número de parámetros del modelo.
            weights (Tuple[float, float, float]): Pesos para precisión,
            pérdida y parámetros; deben sumar 1.

        Retorna:
            float: Puntaje combinado, donde un valor mayor indica mejor desempeño.

        Lanza:
            ValueError: Si la suma de los pesos no es igual a 1 o si los valores
            máximos son menores o iguales a cero.
        """
        # Validación de pesos
        if abs(sum(weights) - 1.0) > EPSILON:
            raise ValueError("Los pesos deben sumar 1.")

        # Validación de valores máximos para evitar divisiones por cero o
        # comportamientos inesperados
        if self.max_precision <= 0 or self.max_loss <= 0 or self.max_params <= 0:
            raise ValueError("Los valores máximos deben ser mayores a cero.")

        peso_precision, peso_loss, peso_params = weights

        # Normalización de métricas a escala [0, 1]
        precision_normalizada = precision / self.max_precision
        loss_normalizada = 1 - (loss / self.max_loss)  # Menor pérdida es mejor
        params_normalizados = 1 - (
            num_params / self.max_params
        )  # Menos parámetros es mejor

        # Cálculo del puntaje combinado ponderado
        puntaje = (
            peso_precision * precision_normalizada
            + peso_loss * loss_normalizada
            + peso_params * params_normalizados
        )

        return puntaje


# Model Builder with Open/Closed principle - extensible for new model types
FIRST_LAYER_MIN_NEURONS = 32
FIRST_LAYER_MAX_NEURONS = 256
OTHER_LAYER_MIN_NEURONS = 5
OTHER_LAYER_MAX_NEURONS = 1024
MIN_LAYERS = 3
MAX_LAYERS = 8


class ModelBuilder:
    """
    Construye un modelo Keras optimizado mediante hiperparámetros sugeridos por Optuna.
    """

    def __init__(self, input_dim: int):
        """
        Inicializa el constructor de modelo con la dimensión de entrada.

        Parámetros:
            input_dim (int): Dimensión de entrada, debe ser un entero positivo.
        """
        assert input_dim > 0, "La dimensión de entrada debe ser un entero positivo."
        self.input_dim = input_dim

    def build_model(self, trial: optuna.Trial) -> Sequential:
        """
        Construye un modelo Keras utilizando el trial de Optuna para definir hiperparámetros.

        Parámetros:
            trial (optuna.Trial): Objeto que sugiere valores de hiperparámetros.

        Retorna:
            Sequential: Modelo Keras construido.
        """
        assert trial is not None, "Se debe proporcionar un trial válido."
        model = Sequential()

        # Primera capa con dimensión de entrada
        self._add_layer(model, trial, layer_idx=0, is_first_layer=True)

        # Capas ocultas según sugerencia de trial
        n_layers = trial.suggest_int("n_layers", MIN_LAYERS, MAX_LAYERS)
        for i in range(1, n_layers):
            self._add_layer(model, trial, layer_idx=i)

        # Capa de salida para clasificación binaria
        model.add(Dense(1, activation="sigmoid"))
        return model

    def _add_layer(
        self,
        model: Sequential,
        trial: optuna.Trial,
        layer_idx: int,
        is_first_layer: bool = False,
    ) -> None:
        """
        Agrega una capa oculta al modelo según los hiperparámetros sugeridos.

        Parámetros:
            model (Sequential): Modelo Keras al que se agregará la capa.
            trial (optuna.Trial): Objeto para sugerir hiperparámetros.
            layer_idx (int): Índice de la capa.
            is_first_layer (bool): Indica si la capa es la primera.
        """
        # Validar argumentos
        assert (
            isinstance(layer_idx, int) and layer_idx >= 0
        ), "El índice de la capa debe ser un entero no negativo."
        assert isinstance(
            model, Sequential
        ), "El modelo debe ser una instancia de Sequential."

        # Definir rango de neuronas según la posición de la capa
        if is_first_layer:
            min_neurons, max_neurons = FIRST_LAYER_MIN_NEURONS, FIRST_LAYER_MAX_NEURONS
        else:
            min_neurons, max_neurons = OTHER_LAYER_MIN_NEURONS, OTHER_LAYER_MAX_NEURONS

        # Sugerir número de neuronas
        n_neurons = trial.suggest_int(
            f"n_neurons_l{layer_idx}", min_neurons, max_neurons
        )
        assert (
            min_neurons <= n_neurons <= max_neurons
        ), "El número de neuronas está fuera del rango permitido."

        # Sugerir función de activación
        activation_options = ["relu", "tanh", "elu", "selu", "leaky_relu", "swish"]
        activation_type = trial.suggest_categorical(
            f"activation_l{layer_idx}", activation_options
        )
        assert (
            activation_type in activation_options
        ), "El tipo de activación no es válido."

        # Agregar capa densa con la configuración adecuada
        if is_first_layer:
            dense_params = {"units": n_neurons, "input_dim": self.input_dim}
        else:
            dense_params = {"units": n_neurons}

        if activation_type == "leaky_relu":
            model.add(Dense(**dense_params))
            alpha = trial.suggest_float(f"alpha_l{layer_idx}", 0.01, 0.3, step=0.01)
            model.add(LeakyReLU(alpha=alpha))
        elif activation_type == "swish":
            dense_params["activation"] = ActivationFunctions.swish
            model.add(Dense(**dense_params))
        else:
            dense_params["activation"] = activation_type
            model.add(Dense(**dense_params))

        # Agregar capa de dropout con hiperparámetro sugerido
        dropout_rate = trial.suggest_float(
            f"dropout_l{layer_idx}", 0.0, 0.5, step=0.001
        )
        model.add(Dropout(dropout_rate))


# Training configuration - Separated for Single Responsibility
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 10
MIN_LEARNING_RATE = 0.0001
MAX_LEARNING_RATE = 0.01
MIN_BATCH_SIZE = 10
MAX_BATCH_SIZE = 512


class TrainingConfig:
    """
    Configuración de entrenamiento que incluye parámetros y callbacks.
    """

    def __init__(
        self, max_epochs: int = DEFAULT_MAX_EPOCHS, patience: int = DEFAULT_PATIENCE
    ):
        """
        Inicializa la configuración con el número máximo de épocas y la paciencia para
        early stopping.

        Parámetros:
            max_epochs (int): Número máximo de épocas, debe ser mayor que 0.
            patience (int): Paciencia para early stopping, debe ser mayor que 0.
        """
        assert max_epochs > 0, "max_epochs debe ser mayor que 0."
        assert patience > 0, "patience debe ser mayor que 0."
        self.max_epochs = max_epochs
        self.patience = patience

    def get_callbacks(self, trial: optuna.Trial) -> List:
        """
        Obtiene callbacks para el entrenamiento, incluyendo early stopping y pruning.

        Parámetros:
            trial (optuna.Trial): Objeto trial para sugerir hiperparámetros.

        Retorna:
            List: Lista de callbacks configurados.
        """
        assert trial is not None, "Se debe proporcionar un trial válido."
        callbacks = [
            EarlyStopping(
                monitor="val_acc", patience=self.patience, restore_best_weights=True
            ),
            TFKerasPruningCallback(trial, monitor="val_accuracy"),
        ]
        assert callbacks, "La lista de callbacks no debe estar vacía."
        return callbacks

    def get_optimizer(self, trial: optuna.Trial) -> Union[Adam, RMSprop]:
        """
        Obtiene el optimizador basándose en las sugerencias del trial.

        Parámetros:
            trial (optuna.Trial): Objeto trial para sugerir hiperparámetros.

        Retorna:
            Union[Adam, RMSprop]: Instancia del optimizador configurado.
        """
        assert trial is not None, "Se debe proporcionar un trial válido."
        learning_rate = trial.suggest_float(
            "lr", MIN_LEARNING_RATE, MAX_LEARNING_RATE, step=MIN_LEARNING_RATE
        )
        assert (
            MIN_LEARNING_RATE <= learning_rate <= MAX_LEARNING_RATE
        ), "El learning rate está fuera de rango."

        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        assert optimizer_name in [
            "Adam",
            "RMSprop",
        ], "El optimizador sugerido no es válido."

        if optimizer_name == "Adam":
            optimizer = Adam(  # pylint: disable=redefined-outer-name
                learning_rate=learning_rate
            )
        else:
            optimizer = RMSprop(learning_rate=learning_rate)

        assert optimizer is not None, "Error al crear el optimizador."
        return optimizer

    def get_batch_size(self, trial: optuna.Trial) -> int:
        """
        Obtiene el tamaño del batch basándose en las sugerencias del trial.

        Parámetros:
            trial (optuna.Trial): Objeto trial para sugerir hiperparámetros.

        Retorna:
            int: Tamaño del batch seleccionado.
        """
        assert trial is not None, "Se debe proporcionar un trial válido."
        batch_size = trial.suggest_int("batch_size", MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        assert (
            MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE
        ), "El tamaño del batch está fuera del rango permitido."
        return batch_size


# Data Handler - Separate class for data preparation
class DataHandler:
    """
    Maneja y procesa los datos de entrenamiento y validación.
    """

    def __init__(self, train_features, val_features, train_labels, val_labels):
        """
        Inicializa el handler convirtiendo los datos a arreglos de NumPy y
        determinando la dimensión de entrada.
        """
        self.x_train = np.array(train_features)
        self.x_val = np.array(val_features)
        self.y_train = np.array(train_labels)
        self.y_val = np.array(val_labels)
        self.input_dim = self.x_train.shape[1]

    def get_dimensions(self) -> Tuple[int, int, int]:
        """
        Retorna las dimensiones de los datos: número de muestras de entrenamiento,
        validación y la cantidad de características.
        """
        num_train_samples = len(self.x_train)
        num_val_samples = len(self.x_val)
        return num_train_samples, num_val_samples, self.input_dim


# Main optimizer class using composition
# Constantes para la optimización
N_TRIALS_DEFAULT = 20
N_JOBS_DEFAULT = 1
STUDY_NAME_DEFAULT = "neural_network_study"
REDUCTION_FACTOR = 3


class NeuralNetworkOptimizer:
    """
    Optimiza una red neuronal para clasificación binaria usando Optuna y TensorFlow.
    """

    def __init__(self, x_train, x_val, y_train, y_val):  # pylint: disable=W0621
        self.data_handler = DataHandler(x_train, x_val, y_train, y_val)
        self.model_builder = ModelBuilder(self.data_handler.input_dim)
        self.training_config = TrainingConfig()
        self.evaluator = ModelEvaluator()
        self.study = None
        self.trial_results = []

    def objective(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para la optimización con Optuna.
        """
        tf.keras.backend.clear_session()  # pylint: disable=E1101
        model = self.model_builder.build_model(trial)

        optimizer = self.training_config.get_optimizer(trial)  # pylint: disable=W0621
        batch_size = self.training_config.get_batch_size(trial)
        callbacks = self.training_config.get_callbacks(trial)

        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["acc"])
        print(f"Trial {trial.number}: {trial.params}")
        model.summary()

        start_time = time.time()
        history = model.fit(
            self.data_handler.x_train,
            self.data_handler.y_train,
            validation_data=(self.data_handler.x_val, self.data_handler.y_val),
            batch_size=batch_size,
            epochs=self.training_config.max_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        training_time = time.time() - start_time

        val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"
        val_accuracy = max(history.history[val_acc_key])
        val_loss = min(history.history["val_loss"])
        num_params = model.count_params()

        score = self.evaluator.calculate_score(val_accuracy, val_loss, num_params)

        trial.set_user_attr("acc", float(val_accuracy))
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("n_params", num_params)
        trial.set_user_attr("training_time", training_time)

        print(
            f"Trial {trial.number} finished: {val_acc_key}={val_accuracy:.4f},time={training_time:.2f}s"  # pylint: disable=C0301
        )
        self.trial_results.append(
            {
                **trial.params,
                "value": score,
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
                "n_params": num_params,
                "training_time": training_time,
                "trial_number": trial.number,
            }
        )
        return score

    def optimize(
        self,
        n_trials: int = N_TRIALS_DEFAULT,
        n_jobs: int = N_JOBS_DEFAULT,
        study_name: str = STUDY_NAME_DEFAULT,
        storage: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Ejecuta la optimización de hiperparámetros y retorna los mejores parámetros y valor.
        """
        sampler = optuna.samplers.TPESampler()
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=self.training_config.max_epochs,
            reduction_factor=REDUCTION_FACTOR,
        )
        self.study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        self.study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
        return self.study.best_params, self.study.best_value

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Retorna los resultados de los trials como un DataFrame.
        """
        if not self.trial_results and self.study:
            for trial in self.study.trials:
                if trial.state.is_finished():
                    self.trial_results.append(
                        {
                            **trial.params,
                            "value": trial.value,
                            "val_accuracy": trial.user_attrs.get("val_accuracy"),
                            "val_loss": trial.user_attrs.get("val_loss"),
                            "n_params": trial.user_attrs.get("n_params"),
                            "training_time": trial.user_attrs.get("training_time"),
                            "trial_number": trial.number,
                        }
                    )
        return pd.DataFrame(self.trial_results)

    def build_best_model(self) -> Sequential:
        """
        Construye el modelo utilizando los mejores hiperparámetros encontrados.
        """
        if not self.study or not self.study.best_params:
            raise ValueError("No se han encontrado parámetros óptimos.")

        class BestTrial:
            """Simula un objeto Trial de Optuna usando los mejores parámetros."""

            def __init__(self, params):
                """Inicializa con un diccionario de parámetros."""
                self.params = params

            def suggest_categorical(self, name, _choices):  # Pylint: disable=W0613
                """Devuelve el valor categórico para 'name'."""
                return self.params[name]

            def suggest_float(
                self, name, _low, _high, _step=None  # Pylint: disable=W0613
            ):
                """Devuelve el valor flotante para 'name'."""
                return self.params[name]

            def suggest_int(self, name, _low, _high):  # Pylint: disable=W0613
                """Devuelve el valor entero para 'name'."""
                return self.params[name]

        best_trial = BestTrial(self.study.best_params)
        return self.model_builder.build_model(best_trial)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate sample binary classification dataset
    X, y = make_classification(
        n_samples=500, n_features=10, n_classes=2, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create optimizer
    optimizer = NeuralNetworkOptimizer(x_train, x_val, y_train, y_val)

    # Run optimization
    best_params, best_value = optimizer.optimize(
        n_trials=20,
        study_name="binary_classification_study",
        storage="sqlite:///neural_network_optimization.db",
    )

    # Print results
    print("\nBest hyperparameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation score: {best_value:.4f}")

    # Get and export results
    results_df = optimizer.get_results_dataframe()
    results_df.to_csv("optimization_results.csv", index=False)
    print("Results exported to 'optimization_results.csv'")

    # Build best model
    best_model = optimizer.build_best_model()
    print("\nBest model summary:")
    best_model.summary()
