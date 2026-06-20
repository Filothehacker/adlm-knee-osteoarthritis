import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dino_v3.data import list_image_files, list_patient_archives
from dino_v3.main import configure_runtime_dirs


def test_configure_runtime_dirs_forces_cache_and_tmp_under_requested_roots(
    monkeypatch, tmp_path
):
    cache_dir = tmp_path / "cache"
    pip_cache_dir = tmp_path / "pip-cache"
    tmp_dir = tmp_path / "tmp"
    monkeypatch.setenv("PIP_CACHE_DIR", str(tmp_path / "home-cache" / "pip"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "home-cache" / "torch"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home-cache" / "hf"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "home-cache"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "bad-tmp"))

    configure_runtime_dirs(str(cache_dir), str(tmp_dir), str(pip_cache_dir))

    assert os.environ["PIP_CACHE_DIR"] == str(pip_cache_dir)
    assert os.environ["TORCH_HOME"] == str(cache_dir / "torch")
    assert os.environ["HF_HOME"] == str(cache_dir / "huggingface")
    assert os.environ["XDG_CACHE_HOME"] == str(cache_dir)
    assert os.environ["TMPDIR"] == str(tmp_dir)
    assert tempfile.tempdir == str(tmp_dir)
    assert (cache_dir / "torch").is_dir()
    assert (cache_dir / "huggingface").is_dir()
    assert pip_cache_dir.is_dir()
    assert tmp_dir.is_dir()


def test_list_image_files_recurses_supported_extensions(tmp_path):
    (tmp_path / "subset" / "a").mkdir(parents=True)
    (tmp_path / "subset" / "b").mkdir(parents=True)
    supported = [
        tmp_path / "subset" / "a" / "one.JPG",
        tmp_path / "subset" / "a" / "two.tif",
        tmp_path / "subset" / "b" / "three.webp",
        tmp_path / "root.png",
    ]
    ignored = [
        tmp_path / "subset" / "a" / "notes.txt",
        tmp_path / "subset" / "b" / "archive.tar.gz",
    ]
    for path in supported + ignored:
        path.write_text("x")

    assert list_image_files(str(tmp_path)) == sorted(str(path) for path in supported)


def test_list_patient_archives_is_sorted_by_patient_id(tmp_path):
    for subset, patient in [("0.E.1", "9002"), ("0.C.2", "9001")]:
        side_dir = tmp_path / subset / patient / "mri" / "left"
        side_dir.mkdir(parents=True)
        (side_dir / "scan.tar.gz").write_text("placeholder")

    assert list_patient_archives(str(tmp_path), "left") == [
        ("9001", str(tmp_path / "0.C.2" / "9001" / "mri" / "left" / "scan.tar.gz")),
        ("9002", str(tmp_path / "0.E.1" / "9002" / "mri" / "left" / "scan.tar.gz")),
    ]
