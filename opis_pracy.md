**„Porównanie metod detekcji i lokalizacji zawodników piłki nożnej na płaszczyźnie boiska w oparciu o obraz transmisji telewizyjnej"**

**Opis pracy** 

Celem pracy jest opracowanie i porównanie metod lokalizacji zawodników piłki nożnej na płaszczyźnie boiska na podstawie obrazów z transmisji telewizyjnej. Zadanie to wymaga rozwiązania dwóch powiązanych problemów: detekcji zawodników na obrazie oraz estymacji homografii umożliwiającej odwzorowanie współrzędnych pikselowych na rzeczywiste pozycje na boisku. 
W ramach pracy zostanie przeprowadzona analiza porównawcza wybranych metod detekcji obiektów (np. z rodziny YOLO, Faster R-CNN lub DETR) oraz podejść do estymacji homografii — zarówno opartych na klasycznej detekcji cech charakterystycznych boiska, jak i wykorzystujących głębokie sieci neuronowe. Zbadany zostanie wpływ różnych kombinacji tych metod na końcową dokładność lokalizacji zawodników w warunkach zmiennej perspektywy kamery, częściowego przysłaniania zawodników oraz zróżnicowanego oświetlenia. 
Wynikiem pracy będzie prototyp systemu realizującego pipeline detekcja–kalibracja–projekcja wraz z ewaluacją ilościową na publicznie dostępnym zbiorze danych. 
 
**Minimalne wymagania**
- Przegląd literatury — omówienie istniejących metod detekcji obiektów i estymacji homografii w kontekście analizy obrazu sportowego 
- Implementacja pipeline'u składającego się z:  
    - co najmniej  metod detekcji zawodników
    - co najmniej dwóch metod estymacji homografii 
- Ewaluacja ilościowa — porównanie kombinacji metod z użyciem odpowiednich metryk (np. błąd reprojekcji, IoU detekcji, błąd lokalizacji w metrach) 
- Analiza jakościowa — dyskusja przypadków trudnych (okluzje, nietypowe ujęcia, słaba widoczność linii boiska) 
- Dokumentacja — opis architektury rozwiązania, instrukcja uruchomienia, repozytorium z kodem
