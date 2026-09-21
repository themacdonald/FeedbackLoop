from pathlib import Path
import json

from feedbackloop.cli import main

def test_cli_demo_writes_verified_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import sys
    monkeypatch.setattr(sys, "argv", ["feedbackloop", "--demo"])
    main()
    artifact = Path("artifacts/work_demo.json")
    data = json.loads(artifact.read_text())
    assert data["state"] == "completed"
    assert len(data["events"]) == 4
