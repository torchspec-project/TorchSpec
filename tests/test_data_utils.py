import os
from pathlib import Path

from torchspec.data import utils as data_utils


def test_load_hf_dataset_treats_existing_relative_dir_as_local(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "my_dataset"
    data_dir.mkdir(parents=True)
    (data_dir / "sample.jsonl").write_text('{"x": 1}\n')

    rel_path = Path(os.path.relpath(data_dir, Path.cwd()))

    called = {}

    def fake_load_dataset(fmt, data_files, split, streaming):
        called["fmt"] = fmt
        called["data_files"] = data_files
        called["split"] = split
        called["streaming"] = streaming
        return "LOCAL_DATASET"

    monkeypatch.setattr(data_utils, "load_dataset", fake_load_dataset)

    out = data_utils.load_hf_dataset(str(rel_path))

    assert out == "LOCAL_DATASET"
    assert called["fmt"] == "json"
    assert called["split"] == "train"
    assert called["streaming"] is True
    assert any(str(f).endswith("sample.jsonl") for f in called["data_files"])
