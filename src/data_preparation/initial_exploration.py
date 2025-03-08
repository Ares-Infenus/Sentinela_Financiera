import pandas as pd
import numpy as np


class InitialExploration:
    """
    Clase para realizar una exploración inicial de un DataFrame.
    Se evalúan aspectos clave y se retorna un diccionario donde algunos
    valores son tuplas que indican (problema_existe, cantidad).
    """

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa la clase con una copia del DataFrame para evitar modificar el original.
        """
        self.df = df.copy()

    def has_info(self) -> bool:
        """
        Verifica si el DataFrame tiene datos (no está vacío).
        """
        return not self.df.empty

    def count_missing_rows(self) -> int:
        """
        Retorna la cantidad de filas que tienen al menos un valor nulo.
        """
        return int(self.df.isnull().any(axis=1).sum())

    def count_duplicate_rows(self) -> int:
        """
        Retorna la cantidad de filas duplicadas.
        """
        return int(self.df.duplicated().sum())

    def count_non_float_columns(self) -> int:
        """
        Retorna la cantidad de columnas que no son de tipo float.
        """
        return sum(
            not pd.api.types.is_float_dtype(self.df[col]) for col in self.df.columns
        )

    def count_infinite_rows(self) -> int:
        """
        Retorna la cantidad de filas que contienen al menos un valor infinito (inf o -inf)
        en columnas numéricas.
        """
        # Solo se consideran columnas numéricas para evitar problemas con strings.
        num_df = self.df.select_dtypes(include=[np.number])
        return int(np.isinf(num_df).any(axis=1).sum())

    def run_all(self) -> dict:
        """
        Ejecuta todas las verificaciones y devuelve un diccionario con los resultados.
        La estructura del diccionario es:
            {
                "inf": bool,                   # True si el DataFrame tiene datos.
                "nan": (bool, int),            # (True, cantidad) si hay filas con nulos, (False, 0) de lo contrario.
                "duplicate": (bool, int),      # (True, cantidad) si hay filas duplicadas, (False, 0) de lo contrario.
                "columnas_no_float": (bool, int), # (True, cantidad) si hay columnas que no son float, (False, 0) de lo contrario.
                "infinite": (bool, int)        # (True, cantidad) si hay filas con inf o -inf, (False, 0) de lo contrario.
            }
        """
        missing_count = self.count_missing_rows()
        duplicate_count = self.count_duplicate_rows()
        non_float_count = self.count_non_float_columns()
        infinite_count = self.count_infinite_rows()

        results = {
            "inf": self.has_info(),
            "nan": (True, missing_count) if missing_count > 0 else (False, 0),
            "duplicate": (True, duplicate_count) if duplicate_count > 0 else (False, 0),
            "columnas_no_float": (
                (True, non_float_count) if non_float_count > 0 else (False, 0)
            ),
            "infinite": (True, infinite_count) if infinite_count > 0 else (False, 0),
        }
        return results


# Ejemplo de uso:
if __name__ == "__main__":
    # Ejemplo de DataFrame con un valor infinito en la columna 'A'
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, np.inf, None],
            "B": [4.0, None, 6.0, 7.0],
            "C": ["x", "y", "y", "z"],
        }
    )

    exploration = InitialExploration(df)
    results = exploration.run_all()
    print(results)
