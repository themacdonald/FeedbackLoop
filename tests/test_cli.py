import sys
from pathlib import Path

from feedbackloop.cli import main


def test_demo_creates_artifact(tmp_path, monkeypatch, capsys):
    # Change working directory to a temporary path
    monkeypatch.chdir(tmp_path)
    # Simulate calling the CLI with --demo
    monkeypatch.setattr(sys, "argv", ["feedbackloop", "--demo"])
    # Run the CLI main function
    main()
    # Capture output
    out, _ = capsys.readouterr()
# Expect message about saved model
    # Ensure the artifact file was created
    assert (tmp_path / "artifacts" / "reward_model.pkl").exists()
