"""Verify local weight/GGUF SHA-256 against cached Hugging Face download ETags.

For large LFS files the 64-hex ETag is the content SHA-256. Unknown ETag formats are not
treated as successful verification. Reads only; does not modify model files.
"""
import hashlib
import json
from pathlib import Path
import re
import argparse


def verify(output=Path("reproductions/model_integrity.json")):
    paths = sorted(Path("models/bggpt-4b").glob("*.safetensors")) + sorted(Path("models/gguf").glob("*.gguf"))
    result = []
    for path in paths:
        metadata = path.parent/".cache/huggingface/download"/(path.name + ".metadata")
        lines = metadata.read_text().splitlines()
        expected = lines[1].strip('"')
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            result.append({"path": str(path), "verified": False, "reason": "ETag is not a content SHA-256"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b""):
                digest.update(chunk)
        row = {"path": str(path), "repository_commit": lines[0], "expected_sha256": expected,
               "actual_sha256": digest.hexdigest(), "verified": digest.hexdigest() == expected}
        result.append(row)
        print(str(path), "VERIFIED" if row["verified"] else "MISMATCH", flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    if not paths or not all(row["verified"] for row in result):
        raise SystemExit("Model integrity not established for every file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("reproductions/model_integrity.json"))
    verify(parser.parse_args().output)
