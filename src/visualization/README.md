# visualization

Moduł odpowiada za przygotowanie graficznej reprezentacji wyników działania pipeline'u.

## `pitch_drawer.py`

Generuje dwuwymiarowy model boiska o wymiarach 105 m × 68 m wraz z podstawowymi oznaczeniami boiskowymi.

Układ współrzędnych jest wycentrowany:

```text
X: -52.5 ... +52.5 m
Y: -34.0 ... +34.0 m
```

## `minimap.py`

Buduje minimapę na podstawie modelu boiska i nanosi na nią pozycje obiektów zwrócone przez etap projekcji.

Minimapa jest następnie łączona z oryginalną klatką zawierającą ramki detekcji, ID tracków oraz pomocnicze oznaczenia drużyn.

Moduł służy przede wszystkim do prezentacji i jakościowej analizy wyników.
