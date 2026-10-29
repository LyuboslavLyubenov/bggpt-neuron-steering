"""Create a fresh experiment workspace without copying the published generated outputs.

Copies source/configuration/calibration only, and links an existing models/ directory. Refuses an
existing destination. The original repository/evidence and model files are not edited.
"""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination",type=Path)
    args = parser.parse_args()
    target = args.destination.expanduser().resolve()
    if target.exists():
        raise SystemExit("Destination exists; choose a new directory. Nothing overwritten.")
    target.mkdir(parents=True)
    for path in ROOT.iterdir():
        if path.is_file() and (path.suffix in (".py",".js",".toml",".lock")
                               or path.name in ("unit.json",".python-version",".gitignore")):
            shutil.copy2(path,target/path.name)
    (target/"presets").mkdir()
    shutil.copy2(ROOT/"presets/evil_chat_bg.json",target/"presets/evil_chat_bg.json")
    (target/"personas/evil_web_tuning").mkdir(parents=True)
    models = ROOT/"models"
    if models.is_dir():
        (target/"models").symlink_to(models.resolve(),target_is_directory=True)
    print(f"Fresh workspace: {target}")
    print("No prior generated outputs or subjective annotations copied. Original unit.json calibration retained.")
    print("If models/ is absent, download models there first. Run uv sync --frozen in the new directory.")


if __name__ == "__main__":
    main()
