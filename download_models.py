"""Download the study's pinned upstream checkpoints; never bypass model access/licence requirements."""
from huggingface_hub import snapshot_download, hf_hub_download

SAFETENSORS_REPO = "INSAIT-Institute/BgGPT-Gemma-3-4B-IT"
SAFETENSORS_REVISION = "7a911c47ed3dd37ad923c5286cf0bfe6a3eea6f2"
GGUF_REPO = "INSAIT-Institute/BgGPT-Gemma-3-4B-IT-GGUF"
GGUF_REVISION = "70a57d6462a25c0599e590bd8ea99a86a1752975"
GGUF_FILE = "BgGPT-Gemma-3-4B-IT-Q4_K_M.gguf"


def main():
    print("Read/accept upstream model terms first; use hf auth login if required.")
    snapshot_download(SAFETENSORS_REPO, revision=SAFETENSORS_REVISION,
                      local_dir="models/bggpt-4b",
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "README.md"])
    hf_hub_download(GGUF_REPO, GGUF_FILE, revision=GGUF_REVISION, local_dir="models/gguf")
    print("Downloaded pinned checkpoints. Run uv run python verify_model_files.py to verify weight hashes.")


if __name__ == "__main__":
    main()
