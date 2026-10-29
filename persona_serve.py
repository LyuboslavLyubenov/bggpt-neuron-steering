"""Launch the local llama.cpp chat UI with a reviewed persona preset.

uv run python persona_serve.py --list
uv run python persona_serve.py poet --port 8801
"""
import argparse
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/"personas/validated_presets.json").read_text())
    presets = manifest["presets"]
    parser = argparse.ArgumentParser()
    parser.add_argument("persona", nargs="?", choices=sorted(presets))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--port", type=int, default=8801)
    args = parser.parse_args()
    if args.list:
        for name, p in presets.items():
            print(name, p["recipe"], p["result"])
        return
    if not args.persona:
        parser.error("Choose a persona or pass --list")
    vector = root/presets[args.persona]["vector"]
    if not vector.exists():
        parser.error("Preset vector missing; run uv run python build_persona_presets.py first")
    command = ["llama-server", "-m", str(root/"models/gguf"/manifest["file"]),
               "--control-vector", str(vector), "--host", "127.0.0.1", "--port", str(args.port),
               "-c", "4096", "-np", "1"]
    print(f"Local chat: http://127.0.0.1:{args.port}", flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
