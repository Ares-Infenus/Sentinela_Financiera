"""
Este código define una clase DataFrameCleaner que limpia un DataFrame de pandas según una
configuración dada. Permite eliminar filas con valores infinitos (np.inf, -np.inf), valores
NaN, y duplicados.

Funcionalidad principal:
Configuración: Se define con un diccionario que indica qué limpieza aplicar y con qué umbrales.
Métodos de limpieza:
_clean_infinite(): Elimina filas con una cantidad de valores infinitos mayor o igual al umbral.
_clean_nan(): Elimina filas con valores NaN según el umbral.
_clean_duplicates(): Elimina filas duplicadas.
Método clean(): Aplica las limpiezas en orden y muestra un resumen con el número de filas
eliminadas.
Uso:
Se crea un DataFrame de prueba con NaN, duplicados e infinitos, y se limpia con la configuración
definida.
"""

import pandas as pd
import numpy as np

# Constantes para configuración predeterminada (si fuera necesario extender)
EMPTY_CONFIG_FLAG = False


class DataFrameCleaner:
    """
    Clase para limpiar un DataFrame de pandas basado en una configuración dada.

    La configuración debe ser un diccionario con la siguiente estructura:
      {
        'inf': bool,             # False indica que se detiene la operación (DataFrame vacío)
        'nan': (bool, int),      # (activar, umbral) para eliminar filas con NaN
        'duplicate': (bool, int),# (activar, _) para eliminar duplicados
        'infinite': (bool, int)  # (activar, umbral) para eliminar filas con valores infinitos
      }
    """

    def __init__(self, cfg: dict):
        self.config = cfg

    def _check_dataframe_empty(self) -> bool:
        """
        Verifica si se debe detener la operación según el parámetro 'inf'.
        """
        return not self.config.get("inf", EMPTY_CONFIG_FLAG)

    def _clean_infinite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina filas con una cantidad de valores infinitos (np.inf o -np.inf)
        mayor o igual al umbral definido en la configuración ('infinite': (True, umbral)).
        """
        if self.config.get("infinite", (False, 0))[0]:
            threshold = self.config["infinite"][1]
            infinite_count = df.isin([np.inf, -np.inf]).sum(axis=1)
            df = df[infinite_count < threshold]
        return df

    def _clean_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina filas que contienen valores NaN.
        Si el umbral es 1 o menor, se eliminan todas las filas con al menos un NaN;
        de lo contrario, se eliminan filas con un número de NaN mayor o igual al umbral.
        """
        if self.config.get("nan", (False, 0))[0]:
            threshold = self.config["nan"][1]
            if threshold <= 1:
                df = df.dropna()
            else:
                df = df[df.isna().sum(axis=1) < threshold]
        return df

    def _clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina filas duplicadas si está configurado ('duplicate': (True, _)).
        """
        if self.config.get("duplicate", (False, 0))[0]:
            df = df.drop_duplicates()
        return df

    def clean(self, df: pd.DataFrame):
        """
        Ejecuta la limpieza del DataFrame aplicando:
          1. Verificación del parámetro 'inf' (si es False, se devuelve un mensaje)
          2. Eliminación de filas con valores infinitos
          3. Eliminación de filas con valores NaN
          4. Eliminación de filas duplicadas

        Imprime un resumen de la cantidad de filas eliminadas en cada etapa.
        """
        if self._check_dataframe_empty():
            mensaje = "El DataFrame está vacío"
            print("\nResumen de limpieza:")
            print(" - Valores infinitos eliminados: 0")
            print(" - Filas con NaN eliminadas: 0")
            print(" - Filas duplicadas eliminadas: 0")
            return mensaje

        stats = {}

        # Limpieza de filas con valores infinitos
        rows_before = len(df)
        df = self._clean_infinite(df)
        stats["infinite_removed"] = rows_before - len(df)

        # Limpieza de filas con valores NaN
        rows_before = len(df)
        df = self._clean_nan(df)
        stats["nan_removed"] = rows_before - len(df)

        # Limpieza de filas duplicadas
        rows_before = len(df)
        df = self._clean_duplicates(df)
        stats["duplicates_removed"] = rows_before - len(df)

        print("\nResumen de limpieza:")
        print(f" - Valores infinitos eliminados: {stats['infinite_removed']}")
        print(f" - Filas con NaN eliminadas: {stats['nan_removed']}")
        print(f" - Filas duplicadas eliminadas: {stats['duplicates_removed']}")

        return df


# Ejemplo de uso y prueba simple
if __name__ == "__main__":
    # Configuración de limpieza
    config = {
        "inf": True,
        "nan": (True, 1),
        "duplicate": (True, 0),
        "infinite": (True, 1),
    }

    # Creación de un DataFrame de ejemplo
    df_ejemplo = pd.DataFrame({"A": [1, np.inf, 3, 4, 4], "B": [1, 2, np.nan, 4, 4]})

    # Inicialización del limpiador y ejecución de la limpieza
    limpiador = DataFrameCleaner(config)
    resultado = limpiador.clean(df_ejemplo)

    print("\nDataFrame resultante:")
    print(resultado)


# Ejemplo de uso
if __name__ == "__main__":
    # DataFrame de ejemplo
    datos = {
        "col1": [1, 1, 2, 3, np.inf, 5, np.nan],
        "col2": [2, 2, 3, 4, -np.inf, 6, 7],
        "col3": ["a", "a", "b", "c", "d", "e", "f"],
    }
    df_original = pd.DataFrame(datos)
    print("DataFrame original:")
    print(df_original)

    # Diccionario de configuración:
    # 'inf': True indica que se procede con la limpieza.
    config = {
        "inf": True,  # Si False, se detiene la operación y se devuelve el mensaje.
        "nan": (True, 1),  # Eliminar filas con al menos 1 valor NaN.
        "duplicate": (True, 0),  # No eliminar duplicados.
        "infinite": (True, 1),  # Eliminar filas con al menos 1 valor infinito.
    }

    cleaner = DataFrameCleaner(config)
    resultado = cleaner.clean(df_original)

    if isinstance(resultado, str):
        print("\nMensaje:")
        print(resultado)
    else:
        print("\nDataFrame limpio:")
        print(resultado)
