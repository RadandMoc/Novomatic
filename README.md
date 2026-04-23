# Rozwiązanie zadania rekrutacyjnego - Novomatic

Niniejsze repozytorium zawiera rozwiązanie zadania rekrutacyjnego dla firmy Novomatic. Główna część analityczna, kod oraz sprawozdanie z wykonanych prac znajdują się w interaktywnym notatniku.

## Pliki projektu

* **`Zadanie_rekrutacyjne.pdf`** - oryginalna treść polecenia i wytyczne do zadania.
* **`task.ipynb`** - główne sprawozdanie i kod rozwiązania (Jupyter Notebook).
* **`pyproject.toml` / `uv.lock` / `.python-version`** - pliki konfiguracyjne środowiska, zarządzane przez menedżer pakietów `uv`.

---

## Szybki start: Odtworzenie środowiska

Projekt wykorzystuje narzędzie **[uv](https://docs.astral.sh/uv/)** do szybkiego i powtarzalnego zarządzania zależnościami oraz wersjami Pythona.

Aby przygotować środowisko lokalne do uruchomienia projektu:

1. Upewnij się, że masz zainstalowane narzędzie `uv`.
2. W głównym katalogu projektu zsynchronizuj środowisko, uruchamiając komendę:

```bash
uv sync
```

To polecenie automatycznie pobierze odpowiednią wersję środowiska Python i zainstaluje wszystkie wymagane pakiety na podstawie pliku `uv.lock`.

## Pobieranie danych (DVC)

Zestawy danych wykorzystywane w projekcie są zbyt duże, aby przechowywać je bezpośrednio w repozytorium Git, dlatego są wersjonowane za pomocą DVC (Data Version Control).

Zdalny magazyn (remote) został udostępniony w formie testowej. Niestety musisz mieć konto w **[google console](https://console.cloud.google.com/)**. Co więcej, **muszę dodać twojego maila ręcznie jako akceptowanego do testów, dlatego jeśli jesteś zainteresowany to napisz do mnie a cię dodam**. Wynika to ze zmian w polityce bezpieczeństwa Google i nie można już udostępniać dowolnej osobie plików z DVC za pomocą dysku google.

Aby pobrać wszystkie niezbędne dane do lokalnego katalogu, uruchom w terminalu:

```bash
dvc pull
```

## Śledzenie eksperymentów (MLFlow)

W trakcie prac nad modelami wykorzystano MLFlow do logowania metryk, parametrów i wyników eksperymentów. Po pobraniu danych z DVC (które zawierają również logi eksperymentów), możesz przejrzeć całą historię uczenia bez konieczności ponownego trenowania modeli.

Aby uruchomić interfejs graficzny MLFlow, wpisz w terminalu:

```bash
mlflow ui
```

Następnie otwórz w przeglądarce wskazany przez terminal adres (zazwyczaj jest to `http://127.0.0.1:5000`).

**Ważne:** nie zadziała ci MLFlow, bez pobrania danych z DVC. Plik bazydanych MLFlow został zapisany na dysku googla, wraz z innymi elementami projektu.