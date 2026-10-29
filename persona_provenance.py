"""Capture configuration, package versions and local HF commit metadata without loading model weights."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


def capture():
    source_names = ["common.py", "persona_sweep.py", "persona_gguf.py", "persona_report.py", "unit.json", "uv.lock"]
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_names}
    metadata = {}
    for root in (Path("models/bggpt-4b"), Path("models/gguf")):
        for path in sorted((root/".cache/huggingface/download").glob("*.metadata")):
            lines = path.read_text().splitlines()
            metadata[str(path.relative_to(root)) + " @ " + str(root)] = lines[:2]
    files = {str(p): p.stat().st_size for root in (Path("models/bggpt-4b"), Path("models/gguf"))
             for p in root.glob("*") if p.is_file()}
    return {"captured_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "packages": {p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "gguf", "numpy", "huggingface-hub")},
            "llama_version": subprocess.run(["llama-cli", "--version"], capture_output=True,
                                              text=True, check=True).stdout.strip(),
            "source_sha256": source_hashes, "huggingface_metadata_commit_etag": metadata,
            "model_file_sizes": files,
            "note": "Recorded during the ongoing sweep. Initial flatterer artifacts predate this capture. "
                    "Model weight files were not modified by the experiments."}


if __name__ == "__main__":
    output = Path("personas/environment.json")
    if output.exists():
        raise SystemExit("Provenance already exists; refusing to silently overwrite it")
    output.write_text(json.dumps(capture(), indent=2))
    print(f"Saved {output}")
