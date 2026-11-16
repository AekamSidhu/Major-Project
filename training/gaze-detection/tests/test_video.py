from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="run download_model.py to test MediaPipe itself")


def write_video(path: Path, seconds: float, fps: float = 30.0) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for i in range(int(seconds * fps)):
        frame = np.full((120, 160, 3), (i * 3) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_samples_at_the_target_rate_and_crops(tmp_path):
    from readgaze.landmarks import FaceTracker
    from readgaze.video import COLUMNS, face_found_fraction, frames_to_trace, video_frames

    video = tmp_path / "blank.avi"
    write_video(video, seconds=2.0)
    with FaceTracker(MODEL) as tracker:
        frames = video_frames(str(video), tracker, fps=10.0, roi=(10, 10, 100, 80))
    assert list(frames.columns) == COLUMNS
    assert len(frames) == pytest.approx(20, abs=1)
    assert face_found_fraction(frames) == 0.0  # no face in a blank video
    trace = frames_to_trace(frames)
    assert trace.valid_fraction == 0.0


def test_extract_video_script_runs(tmp_path, capsys):
    import extract_video

    video = tmp_path / "blank.avi"
    write_video(video, seconds=12.0)
    extract_video.main([str(video), "--out", str(tmp_path / "frames.csv")])
    assert (tmp_path / "frames.csv").exists()
    assert "not enough valid gaze" in capsys.readouterr().out
