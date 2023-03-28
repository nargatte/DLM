# Deep Learning Methods, Projekt 1, Marcin Galiński, Grzegorz Krzysiak

## Wymagania
Projekt został napisany w języku Python 3.10.9. Z wyjątkiem pliku `AUC.py`, w którym występują anotacje typów (type annotations), nie są używane żadne szczególne funkcjonalności nowszych wersji Pythona, więc inne wersje interpretera mogą, ale nie muszą być kompatybilne z kodem.

Pakiety niezbędne do uruchomienia projektu zostały wylistowane w pliku `requirements.txt`. Zamieszczone zostały wersje użyte podczas implementacji - ponownie, inne wersje mogą, ale nie muszą, być kompatybilne z kodem. Pakiety można zainstalować korzystając z polecenia `pip3 install -r requirements.txt`.

## Uruchomienie
Punktem wejściowym projektu jest plik `main.py`. Można go uruchomić za pomocą polecenia `python3 main.py` lub `./main.py`. Drugi wariant działa wyłącznie w systemie Linux (w tym w środowisku WSL na Windows 10/11) i wymaga ustawienia uprawnień do wykonywania pliku (np. `chmod 755 main.py`).

## Składowe projektu
Projekt składa się z 4 plików:
* `AUC.py` - zmodyfikowana kopia pliku źródłowego klasy [`ROC_AUC`](https://pytorch.org/ignite/generated/ignite.contrib.metrics.ROC_AUC.html) z pakietu `ignite`. Różnica polega na zmianie wywołania funkcji `roc_auc_score`, umożliwiając obliczanie metryki AUC dla klasyfikacji wieloklasowej,
* `networks.py` - implementacja testowanych modeli,
* `utils.py` - implementacja pomocniczych funkcji służących do ładowania danych oraz uruchamiania testów,
* `main.py` - punkt wejściowy programu, wykorzystuje klasy i funkcje zdefiniowane w pozostałych plikach do ułożenia i uruchomienia testów.

## Zaimplementowane modele
Zaimplementowane zostały 4 modele konwolucyjnych sieci neuronowych:
* `SimpleNet` - prosta sieć, składająca się tylko z warstw konwolucyjnych i pełnych. Implementacja pozwala na łatwe dostosowywanie parametrów sieci poprzez przekazanie odpowiednich tablic w konstruktorze. Achitektura sieci zawsze jest analogiczna: najpierw występują wszystkie warstwy konwolucyjne (po każdej warstwie konwolucyjnej aplikowana jest funkcja ReLU), następnie następuje spłaszczenie tensorów, a po nim wszystkie warstwy pełne, wykorzystujące funkcję ReLU. Konstruktor przyjmuje trzy tablice:
  * `conv_channels` - $n+1$ wartości oznaczających liczby kanałów w warstwach konwolucyjnych, gdzie $n$ jest liczbą pożądanych warstw konwolucyjnych. Dla $i$-tej warstwy $i$-ta oraz $i+1$-sza wartość z tablicy oznaczają odpowiednio jej liczbę kanałów wejściowych i wyjściowych,
  * `kernel_sizes` - $n$ wartości oznaczających rozmiary *kerneli* w warstwach konwolucyjnych, gdzie $n$ jest liczbą pożądanych warstw konwolucyjnych. Dla $i$-tej warstwy $i$-ta wartość z tablicy oznacza jej rozmiar *kernela*,
  * `fc_sizes` - $n+1$ wartości oznaczających liczbę neuronów w warstwach pełnych, gdzie $n$ jest liczbą pożądanych warstw pełnych. Dla $i$-tej warstwy $i$-ta oraz $i+1$-sza wartość z tablicy oznaczają odpowiednio jej liczbę neuronów wejściowych i wyjściowych.
* `PoolingNet` - prosta modyfikacja modelu `SimpleNet` dodająca możliwość dodania warstwy *max-pooling* po każdej warstwie konwolucyjnej. Architektura i konstruktor są analogiczne jak w przypadku modelu `SimpleNet`, przy czym `fc_sizes` jest tutaj czwartym parametrem konstruktora, a trzecim jest tablica `pools` - $n$ wartości logicznych oznaczających, czy po danej warstwie konwolucyjnej ma występować warstwa *max-pooling*. Warstwa ta jest aplikowana przed funkcją ReLU.
* `ResidualNet` - model wykorzystujący bloki rezydualne, wprowadzone w modelu [ResNet](https://ieeexplore.ieee.org/document/7780459). Implementacja ta jest oparta o architekturę *resnet 9* z [niniejszego](https://medium.com/analytics-vidhya/resnet-10f4ef1b9d4c) artykułu na medium.com. Ten model nie jest dostosowywalny - w celu uzyskania innej architektury należy zmodyfikować implementację w pliku `networks.py`.
* `InceptionNet` - model wykorzystujący moduły *inception*, wprowadzone w modelu [GoogLeNet](https://arxiv.org/pdf/1409.4842v1.pdf). Implementacja jest oparta o [niniejszą](https://pytorch.org/vision/main/_modules/torchvision/models/inception.html) implementację dostępną w pakiecie `torchvision`. Podobnie jak w przypadku modelu `ResidualNet`, ten model nie jest dostosowywalny - w celu uzyskania innej architektury należy zmodyfikować implementację w pliku `networks.py`.

## Parametry
Program nie przyjmuje żadnych danych z zewnątrz - wszystkie parametry są ustawiane w kodzie. Poniżej znajduje się lista parametrów wraz z ich wartościami oraz miejscami, w których można je edytować:
* wielkość *batcha* - wynosi 64 i jest zapisana w zmiennej `BATCH_SIZE`, ustawianej w linii 15 w pliku `main.py`; wielkość *batcha* dla wzbogaconego zbioru wynosi `2 * BATCH_SIZE` i jest podana jako argument funkcji `get_loaders_augmented` w linii 91 w pliku `main.py`,
* wielkość komitetu - wynosi 5 i jest zapisana w zmiennej `COMMITTEE_SIZE`, ustawianej w linii 16 w pliku `main.py`,
* ziarno generatora liczb pseudolosowych - w trakcie testów było zmieniane i wynosiło 0, 1 lub 2; ziarno jest ustawiane za pomocą instrukcji `torch.manual_seed(<seed>)`. Instrukcja ta pojawia się w pliku `main.py` przed każdym wywołaniem funkcji `get_loaders_single_model`, `get_loaders_committee` lub `get_loaders_augmented` oraz na początku każdego testu (tj. przed każdym tworzeniem modelu),
* maksymalna liczba epok - wynosi 100 i jest podawana jako wartość stała w liniach 215 i 260 w pliku `utils.py`,
* kąt obrotu obrazu przy rotacji - jest wartością losową z przedziału [1, 180), losowaną w linii 47 w pliku `utils.py`,
* parametry rozmycia gaussowskiego - *kernel* ma wielkość 3x3, a sigma wynosi (1.5, 1.5). Wartości te są podane wprost w linii 54 w pliku `utils.py`

## Tworzenie testów
Do utworzenia testów potrzebne są 4 rzeczy: *data loader* dostarczający dane uczące (lub ich wiele w przypadku komitetu sieci), *data loader* dostarczający dane testowe, testowany model (lub ich wiele w przypadku komitetu sieci) oraz odpowiednia funkcja wywołująca testy.

Do uzyskania *data loaderów* służą następujące funkcje:
* `get_loaders_single_model` - jej przeznaczeniem jest pozyskanie *data loaderów* do testów pojedynczego modelu. Przyjmuje jako parametr rozmiar *batcha* i zwraca pojedynczy *data loader* dostarczający dane uczące oraz *data loader* dostarczający dane testowe.
* `get_loaders_committee` - jej przeznaczeniem jest pozyskanie *data loaderów* do testów komitetu sieci. Przyjmuje jako parametr rozmiar *batcha* oraz rozmiar komitetu i zwraca tablicę *data loaderów* dostarczających dane uczące oraz *data loader* dostarczający dane testowe. Liczba zwróconych *loaderów* zapewniających dane uczące wynosi tyle, ile podany rozmiar komitetu.
* `get_loaders_augmented` - jej przeznaczeniem jest pozyskanie *data loaderów* do testów pojedynczego modelu przy użyciu augmentacji danych. Przyjmuje jako parametr rozmiar *batcha* i zwraca pojedynczy *data loader* dostarczający dane uczące oraz *data loader* dostarczający dane testowe.

Do uruchomienia testów służą następujące funkcje:
* `run_model` - służy do wykonywania testów pojedynczego modelu. Przyjmuje kolejno:
  * testowany model,
  * *data loader* zapewniający dane uczące,
  * *data loader* zapewniający dane testowe,
  * urządzenie, na którym mają być wykonywane obliczenia,
  * wartość logiczną określającą, czy wyświetlić macierz pomyłek po zakończeniu uczenia,
  * wartość logiczną określającą, czy zapisać macierz pomyłek do pliku po zakończeniu uczenia,
  * nazwę pliku, pod jaką zapisać macierz pomyłek,
  * identyfikator uruchomienia, wykorzystywany przez `TensorBoard`.
* `run_models` - służy do wykonywania testów komitetu modeli. Przyjmuje takie same parametry jak funkcja `run_model`, z tą różnicą, że pierwszym parametrem powinna być lista modeli, a drugim lista *data loaderów* zapewniających dane uczące. Rozmiary tych list muszą być równe.

Implementacja umożliwa tworzenie heterogenicznych komitetów modeli, ale nie zostało to wykorzystane w testach. Przykładowy test:

```python
torch.manual_seed(0)                                                 # ustawienie ziarna dla deterministycznych wyników
train_loader, test_loader = get_loaders_augmented(BATCH_SIZE * 2)    # pozyskanie data loaderów
model = ResidualNet()                                                # utworzenie modelu
run_model(model, train_loader, test_loader,                          # uruchomienie testów
    draw=False, save=True, savefile="ResidualNet_Augmented", run_id=RUN_ID)
```

