from sklearn.model_selection import train_test_split


class DataFrameSplitter:
    """
    Clase para separar aleatoriamente un DataFrame en conjuntos de entrenamiento y test,
    extrayendo la columna objetivo según el índice proporcionado.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos.
    target_index : int
        Índice (basado en 0) de la columna que contiene el objetivo.
    test_size : float, opcional (por defecto=0.4)
        Proporción de datos a asignar para el conjunto de test.
    random_state : int, opcional (por defecto=42)
        Semilla para reproducir la aleatoriedad de la división.
    """

    def __init__(self, df, target_index, test_size=0.4, random_state=42):
        self.df = df
        self.target_index = target_index
        self.test_size = test_size
        self.random_state = random_state
        self._split()

    def _split(self):
        # Extraer la columna objetivo y las características.
        # Se elimina la columna objetivo de X para no incluirla en el entrenamiento.
        self.y = self.df.iloc[:, self.target_index]
        self.X = self.df.drop(self.df.columns[self.target_index], axis=1)

        # División en conjuntos de entrenamiento y test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def get_splits(self):
        """
        Devuelve los conjuntos de datos resultantes de la división.

        Retorna:
        --------
        X_train, X_test, y_train, y_test
        """
        return self.X_train, self.X_test, self.y_train, self.y_test


# Ejemplo de uso:
# splitter = DataFrameSplitter(mi_dataframe, target_index=30)
# X_train, X_test, y_train, y_test = splitter.get_splits()
