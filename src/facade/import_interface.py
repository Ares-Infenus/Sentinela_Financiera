import pandas as pd


def IMPORT_FACADE(option):
    if option == "data_raw":
        return pd.read_csv(r"data\raw\creditcard_2023.csv")
    else:
        return None  # o manejo de otros casos
