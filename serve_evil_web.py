"""Launch a specified, evidence-recorded evil web configuration on localhost.

Example: uv run python serve_evil_web.py --config personas/evil_web_tuning/balanced_candidate.json
Browser overrides can supersede server settings; use Reset settings after changing configuration.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FLAGS = {"temperature":"--temp","repeat_penalty":"--repeat-penalty","repeat_last_n":"--repeat-last-n",
         "dry_multiplier":"--dry-multiplier","dry_base":"--dry-base",
         "dry_allowed_length":"--dry-allowed-length","dry_penalty_last_n":"--dry-penalty-last-n"}


def command(config,port):
    vector = ROOT/config["vector"]
    if not vector.is_file():
        raise FileNotFoundError(vector)
    cmd = ["llama-server","-m",str(ROOT/"models/gguf/BgGPT-Gemma-3-4B-IT-Q4_K_M.gguf"),
           "--control-vector-scaled",f"{vector}:{config.get('scale',1.)}",
           "--host","127.0.0.1","--port",str(port),"-c","4096","-np","1",
           "--predict","512"]
    for key,value in config["settings"].items():
        if key not in FLAGS:
            raise ValueError(f"Unsupported sampler setting: {key}")
        cmd.extend([FLAGS[key],str(value)])
    ui = {**config["settings"],"systemMessage":config.get("system",""),"max_tokens":512,
          "enableContinueGeneration":False,"customJson":json.dumps({"tools":[]})}
    cmd += ["--ui-config",json.dumps(ui,ensure_ascii=False)]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--port",type=int,default=8801)
    parser.add_argument("--print-command",action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cmd = command(config,args.port)
    if args.print_command:
        import shlex
        print(shlex.join(cmd))
        return
    print(f"Serving {args.config} at http://127.0.0.1:{args.port}; generation cap is a guardrail, not proof of non-looping.",flush=True)
    os.execvp(cmd[0],cmd)


if __name__ == "__main__":
    main()
