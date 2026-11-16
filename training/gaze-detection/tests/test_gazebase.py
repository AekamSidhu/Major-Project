import io
import zipfile

import numpy as np

from readgaze.datasets.gazebase import RecordingId, iter_recordings, load_csv

CSV = "n,x,y,val,xT,yT,dP,lab\n0,1.0,2.0,0,,,1000,1\n1,1.5,2.5,0,,,1000,1\n2,,,4,,,0,-1\n3,2.0,3.0,0,,,1000,1\n"


def test_parses_recording_names():
    rec = RecordingId.parse("Round_1/Subject_1001/S1/S1_Text/S_1001_S1_TEX.csv")
    assert rec == RecordingId(round=1, participant=1, session=1, task="TEX")
    assert rec.is_reading
    assert RecordingId.parse("S_9335_S2_VD1.csv") == RecordingId(9, 335, 2, "VD1")
    assert RecordingId.parse("notes.csv") is None


def test_loads_seconds_and_marks_invalid_samples():
    trace = load_csv(CSV.encode())
    np.testing.assert_allclose(trace.t, [0.0, 0.001, 0.002, 0.003])
    assert np.isnan(trace.x[2]) and np.isnan(trace.y[2])
    assert trace.x[1] == 1.5


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_reads_nested_zips_without_extracting(tmp_path):
    # Same shape as the real download: the top-level zip holds one zip per participant.
    subject = _zip({"S1/S1_Text/S_1001_S1_TEX.csv": CSV.encode(), "S1/S1_Video_1/S_1001_S1_VD1.csv": CSV.encode()})
    top = tmp_path / "GazeBase_v2_0.zip"
    top.write_bytes(_zip({"Round_1/Subject_1001.zip": subject}))

    found = [rec for rec, _ in iter_recordings(top)]
    assert {r.task for r in found} == {"TEX", "VD1"}

    only_reading = [rec for rec, _ in iter_recordings(top, tasks=["tex"])]
    assert [r.task for r in only_reading] == ["TEX"]


def test_reads_an_extracted_folder(tmp_path):
    folder = tmp_path / "Round_2" / "Subject_2001" / "S2"
    folder.mkdir(parents=True)
    (folder / "S_2001_S2_FXS.csv").write_text(CSV)
    (tmp_path / "Round_2" / "Subject_2002.zip").write_bytes(_zip({"S1/S_2002_S1_TEX.csv": CSV.encode()}))
    found = sorted((r.participant, r.task) for r, _ in iter_recordings(tmp_path))
    assert found == [(1, "FXS"), (2, "TEX")]
