import pandas as pd

print("Wczytywanie danych z pliku JSON + transpozycja...")
dataset_oryginal = pd.read_json("./archive/games.json").transpose()
dataset_oryginal.columns = dataset_oryginal.columns.astype(str)

dataset_oryginal = dataset_oryginal.apply(pd.to_numeric, errors="ignore")

if "release_date" in dataset_oryginal.columns:
    dataset_oryginal["release_date"] = pd.to_datetime(
        dataset_oryginal["release_date"], errors="coerce"
    )

kolumny_object = dataset_oryginal.select_dtypes(include=["object"]).columns
for col in kolumny_object:
    dataset_oryginal[col] = dataset_oryginal[col].astype("string")

print("Zapisywanie do pliku Parquet...")
dataset_oryginal.to_parquet("./archive/games.parquet")

print("Gotowe!")
