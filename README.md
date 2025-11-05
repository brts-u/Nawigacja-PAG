# Nawigacja-PAG
Projekt na Programowanie Aplikacji Geoinformacyjnych \
Wybrany temat: 5 - trasy specjalne \
Autorzy: Bartosz Urbanek, Dominika Zarzycka, Maciej Stawczyk
## Założenia projektu
Celem jest stworzenie aplikacji pozwalającej na wybranie punktu startowego, punktów pośrednich i uzyskanie dobrej trasy łączącej wszystkie punkty i wracającej do punktu startowego (rozwiązanie tzw. problemu komiwojażera).
## Dane wejściowe i wynikowe 
### Danymi wejściowymi są:
- plik/pliki .shp z danymi SKJZ BDOT10k (do utworzenia struktury grafu)
- plik .shp z danymi punktów pośrednich
### Dane wynikowe:
- plik .shp z trasą odtworzona ze źródłowego pliku .shp
# Moduły python
1. shp_graph.py — główny moduł obsługujący strukturę grafu i tworzenie go z plików .shp
2. drawing_plt.py — moduł obsługujący rysowanie grafu, tras i punktów za pomocą matplotlib.pyplot
3. supporting_functions.py — moduł zawierający funkcje pomocnicze wykorzystywane w innych modułach
4. ... uważam, że przydałoby się również rozbić moduł główny na mniejsze części, np. moduł z losowaniem tych tras, albo moduł dijkstry/A*
# Działanie programu (na razie)
__Do uruchomienia programu potrzebne są pliki shapefile z aktualnego BDOT10k_SKJZ__
## Testowanie podstawowych funkcji
Demo w drawing_plt.py oblicza trasę pomiędzy dwoma losowo wybranymi punktami na grafie, a następnie rysuje ją na wykresie.
## Trasy specjalne
nie wiem, może kiedyś...
## Leaflet????
