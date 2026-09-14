# classification

Moduł odpowiada za pomocnicze przypisanie wykrytych zawodników do jednej z dwóch drużyn.

## `team_classifier.py`

Dla każdej detekcji klasy `player` analizowana jest górna część ramki ograniczającej, odpowiadająca w przybliżeniu obszarowi koszulki. Obraz jest przeliczany do przestrzeni HSV, a średnie wartości składowych koloru są następnie grupowane metodą K-means dla `k=2`.

Otrzymany identyfikator drużyny jest wykorzystywany głównie podczas wizualizacji wyników na obrazie oraz minimapie.

Mechanizm ma charakter pomocniczy i nie stanowi części głównego porównania badawczego. W szczególnych przypadkach, takich jak odmienny strój bramkarza lub niejednoznaczne kolory, klasyfikacja może nie odpowiadać rzeczywistej przynależności drużynowej.
