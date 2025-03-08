import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from data_preparation.initial_exploration import InitialExploration
from data_preparation.data_cleaner import DataFrameCleaner
from data_preparation.test_train import DataFrameSplitter
from facade.import_interface import IMPORT_FACADE
from sklearn.model_selection import train_test_split

# TODO: Falta por arreglar la estetica de todos los modulos
# Todo: Falta por documentar todo correctamente
# Todo: hay que construir la ia y las demas formas para poder
# Todo: hay que hacer un articulo cientifico en la latex o por lo menos un informe bien profesional
# Todo: se tiene que hacer una comparacion rigurosa y detallada para saber cuales solas mejores condiguraciones para perfeccioanr cada modelo de prediccion
# Todo: hay que hacer graficas que sean amigables y que ayuden al lector a entender los datos.
# Todo: hay que hacer un reporte en tableau para poder evolucionar y por lo menos aprender algo porque esto no me ayudara a evolucionar.necesito seguir con el proyecto hermes
# todo: se me olvido mencionar la utilizacion de test para probarlos.
# todo: se debe eliminar la seccion de la IA en main pues debe respetar la modularidd para que sea escalable.

if __name__ == "__main__":
    # importacion de datos sin procesar
    data = IMPORT_FACADE(option="data_raw")
    print(data)  # Eliminar despues
    # exploracion inicial
    exploration = InitialExploration(data)
    results = exploration.run_all()
    print(results)  # Eliminar despues
    # Limpieza de datos
    cleaner = DataFrameCleaner(results)
    resultado = cleaner.clean(data)

    if isinstance(resultado, str):
        print("\nMensaje:")
        print(resultado)
    else:
        print("\nDataFrame limpio:")
        print(resultado)

    # viendo el fomato de este
    resultado.info()  # se llego a la conclusion de que no hay necesidad de normalizar pues todas tienen el dtype correspindiente

    # IA
    # Suponiendo que tu DataFrame se llama 'df'
    data_processed = DataFrameSplitter(resultado, target_index=30)
    X_train, X_test, y_train, y_test = data_processed.get_splits()

    # Definir la arquitectura del modelo
    model = Sequential()

    # Capa de entrada y primera capa oculta
    model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dropout(0.5))  # Ayuda a prevenir el sobreajuste

    # Segunda capa oculta
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))

    # Capa de salida para clasificación binaria (0: no fraude, 1: fraude)
    model.add(Dense(1, activation="sigmoid"))

    # Compilar el modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Entrenar el modelo
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # Evaluar el modelo en el conjunto de prueba
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Pérdida en test:", score[0])
    print("Exactitud en test:", score[1])
