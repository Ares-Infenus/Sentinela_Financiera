import pandas as pd


def IMPORT_FACADE(option):
    match option:
        case "data_raw":
            return pd.read_csv(r"data\raw\creditcard_2023.csv")
        case _:
            return None  # o manejo de otros casos
