"""
Plik: src/tracking/bytetrack_tracker.py

Opis:
    Wrapper nad implementacja ByteTrack (Zhang et al., ECCV 2022)
    dostarczana przez biblioteke supervision (Roboflow).

    ByteTrack - dwustopniowy MOT (Multi-Object Tracking):
        1. Detekcje wysokiej pewnosci matchowane sa do istniejacych
           trackow przez IoU + filter Kalmana.
        2. Detekcje niskiej pewnosci uzywane do "ratowania" trackow
           ktore w danej klatce zostaly tymczasowo zgubione (okluzja).

    Tracker oczekuje detekcji w formacie sv.Detections z ustawionym
    polem class_id (klasy traktowane sa oddzielnie - track ID jest
    spojny w obrebie klasy, ale ball ma swoja pule ID, player swoja).

Uzycie:
    tracker = ByteTrackTracker(frame_rate=25)
    for frame in frames:
        detections = detector.detect(frame)          # sv.Detections
        tracked   = tracker.update(detections)        # + tracker_id
        if frame_changed_scene:
            tracker.reset()
"""

import supervision as sv


class ByteTrackTracker:
    """
    Wrapper nad sv.ByteTrack.

    Parametry:
        frame_rate          : klatki na sekunde zrodlowego wideo
        track_thresh        : minimum conf dla rozpoczecia nowego tracka [0..1]
        match_thresh        : minimum IoU dla matchowania detekcji do tracka [0..1]
        lost_track_buffer   : ile klatek tracker pamieta zgubiony track
                              przed jego usunieciem
    """

    def __init__(
        self,
        frame_rate: int = 25,
        track_thresh: float = 0.25,
        match_thresh: float = 0.8,
        lost_track_buffer: int = 30,
    ):
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.lost_track_buffer = lost_track_buffer

        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Aktualizuje stan trackera nowymi detekcjami z biezacej klatki.
        Zwraca te same detekcje uzupelnione o pole tracker_id.
        Detekcje niezmatchowane do zadnego tracka otrzymuja tracker_id = -1.
        """
        return self._tracker.update_with_detections(detections)

    def reset(self) -> None:
        """
        Reset stanu trackera. Wywolywac przy zmianie klipu wideo,
        zeby ID nie przenosily sie miedzy roznymi scenami.
        """
        self._tracker.reset()