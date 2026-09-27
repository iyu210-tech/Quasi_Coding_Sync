"Ensure never delete, overwrite, reuse output tree"

from __future__ import annotations

import pytest

from fibonacci_qc import config as cfg

@pytest.fixture(autouse=True)
def isolated_output_tree(tmp_path,monkeypatch):
    root=tmp_path/"project"
    monkeypatch.setattr(cfg,"PROJECT_ROOT",root)
    monkeypatch.setattr(cfg,"OUTPUT_ROOT",root/"output")
    for name,part in [("OUTPUT_RAW","raw"),("OUTPUT_ANALYSIS","analysis"),("OUTPUT_FIGURES","figures"),("OUTPUT_METADATA","metadata")]:
        monkeypatch.setattr(cfg,name,cfg.OUTPUT_ROOT/part)
    monkeypatch.setattr(cfg,"OUTPUT_DIRS",(cfg.OUTPUT_RAW,cfg.OUTPUT_ANALYSIS,cfg.OUTPUT_FIGURES,cfg.OUTPUT_METADATA))
    cfg.ensure_output_dirs()