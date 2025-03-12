import pandas as pd
import numpy as np


class ManualDataFrameSplitter:
    """
    Clase para dividir manualmente un DataFrame en conjuntos de entrenamiento, validación y test
    según una división predefinida. Solo se requieren los siguientes parámetros:

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos.
    target_index : int
        Índice (basado en 0) de la columna objetivo.
    divition : str
        Cadena que define la división de datos, con las siguientes opciones:
            - "3-3.5-3.5" => train: 30%, test: 35%, validation: 35%
            - "4-3-3"     => train: 40%, test: 30%, validation: 30%
            - "5-2.5-2.5" => train: 50%, test: 25%, validation: 25%
            - "6-2-2"     => train: 60%, test: 20%, validation: 20%
            - "7-1.5-1.5" => train: 70%, test: 15%, validation: 15%
    random_state : int, opcional (por defecto=42)
        Semilla para asegurar la reproducibilidad en la selección aleatoria.
    """

    def __init__(self, df, target_index, divition, random_state=42):
        # Definir las proporciones según el valor de "divition"
        division_map = {
            "3-3.5-3.5": {"train": 0.30, "test": 0.35, "validation": 0.35},
            "4-3-3": {"train": 0.40, "test": 0.30, "validation": 0.30},
            "5-2.5-2.5": {"train": 0.50, "test": 0.25, "validation": 0.25},
            "6-2-2": {"train": 0.60, "test": 0.20, "validation": 0.20},
            "7-1.5-1.5": {"train": 0.70, "test": 0.15, "validation": 0.15},
        }

        if divition not in division_map:
            raise ValueError(
                f"El valor de 'divition' debe ser uno de {list(division_map.keys())}. Recibido: {divition}"
            )

        # Asignar porcentajes
        self.train_pct = division_map[divition]["train"]
        self.test_pct = division_map[divition]["test"]
        self.val_pct = division_map[divition]["validation"]

        total_pct = self.train_pct + self.test_pct + self.val_pct
        # Verificación redundante ya que sabemos que suman 1, pero se puede mantener para seguridad.
        if abs(total_pct - 1) > 1e-6:
            raise ValueError(
                f"La suma de los porcentajes debe ser igual a 1. Recibido: {total_pct}"
            )

        self.df = df
        self.target_index = target_index
        self.random_state = random_state
        self._split()

    def _split(self):
        # Extraer la columna objetivo y las características.
        self.y = self.df.iloc[:, self.target_index]
        self.X = self.df.drop(self.df.columns[self.target_index], axis=1)

        # Número total de muestras
        n = len(self.df)

        # Crear un arreglo de índices y mezclarlo
        np.random.seed(self.random_state)
        indices = np.arange(n)
        np.random.shuffle(indices)

        # Calcular los cortes de índices según los porcentajes
        train_end = int(self.train_pct * n)
        val_end = train_end + int(self.val_pct * n)

        # Extraer índices para cada conjunto
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end : val_end + int(self.test_pct * n)]

        # Asignar los conjuntos usando .iloc
        self.X_train = self.X.iloc[train_indices]
        self.y_train = self.y.iloc[train_indices]
        self.X_val = self.X.iloc[val_indices]
        self.y_val = self.y.iloc[val_indices]
        self.X_test = self.X.iloc[test_indices]
        self.y_test = self.y.iloc[test_indices]

    def get_splits(self):
        """
        Retorna:
        --------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        )


# Ejemplo de uso:
# Supongamos que se desea:
# - Usar la división "4-3-3" => train: 40%, test: 30%, validation: 30%
# splitter = ManualDataFrameSplitter(mi_dataframe, target_index=30, divition="4-3-3")
# X_train, X_val, X_test, y_train, y_val, y_test = splitter.get_splits()
