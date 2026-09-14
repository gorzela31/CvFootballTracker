# tracking

Moduł zawiera pomocniczy mechanizm śledzenia obiektów pomiędzy kolejnymi klatkami transmisji.

## `bytetrack_tracker.py`

Wrapper nad implementacją ByteTrack dostępną w bibliotece `supervision`.

Moduł przyjmuje wyniki detekcji w formacie `supervision.Detections` i uzupełnia je o identyfikatory `tracker_id`, dzięki którym ten sam obiekt może być powiązany pomiędzy następującymi po sobie klatkami.

Tracking jest wykorzystywany we wszystkich czterech pipeline'ach, ale pełni funkcję pomocniczą. Nie stanowi osobnej metody poddanej ewaluacji porównawczej w pracy magisterskiej.
