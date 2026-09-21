import json
import sys

from feedbackloop.cli import main


def test_demo_creates_verifiable_artifact(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["feedbackloop", "--demo"])

    main()

    output = tmp_path / "artifacts" / "work_demo.json"
    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verification"]["valid"] is True
    assert payload["work"]["state"] == "completed"

    out, _ = capsys.readouterr()
    assert "Wrote:" in out
