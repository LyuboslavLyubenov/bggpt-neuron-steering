"""Convert playwright-cli JSON results into a raw browser evidence artifact.

Playwright response.text() used a Latin-1/Windows-1252 interpretation for an SSE endpoint without a
charset. The browser renders Bulgarian correctly; preserve the original log and recover UTF-8 only
for those reconstructed response strings. body().toString('utf8') also exhibited this in the CLI;
DOM innerText was separately verified as correct Bulgarian.
"""
import argparse
import json
from pathlib import Path


def recover_sse_utf8(text):
    if "Ð" not in text and "Ñ" not in text:
        return text
    raw = b"".join(bytes([ord(c)]) if ord(c)<256 else c.encode("cp1252") for c in text)
    return raw.decode("utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    line = args.log.read_text().split("### Result\n",1)[1].splitlines()[0]
    rows = json.loads(line)
    for row in rows:
        row["output"] = recover_sse_utf8(row["output"])
    data = {"source_log":str(args.log),"scope":"actual browser requests including frontend tools/defaults",
            "outputs":rows,"encoding_note":"raw log retained; SSE mojibake recovered only where present"}
    args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(len(rows),"browser replies;",sum(r["finish_reason"]=="stop" for r in rows),"natural stops")


if __name__ == "__main__":
    main()
