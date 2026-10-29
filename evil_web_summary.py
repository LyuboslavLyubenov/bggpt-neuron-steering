"""Summarise operational signals without calling them semantic/persona successes."""
from collections import Counter
import json
from pathlib import Path
import re

from evil_web_tune import OUT
from evil_bulgarian_regression import flags


def summary(rows):
    return {"outputs":len(rows),
            "natural_stops":sum(r["finish_reason"]=="stop" for r in rows),
            "token_limit_or_other_stop":sum(r["finish_reason"]!="stop" for r in rows),
            "repetition_flags":sum(r.get("repeat_review_flag",r.get("loop_proxy",False)) for r in rows),
            "below_80pct_cyrillic":sum(r.get("cyrillic_letter_fraction",1)<.8 for r in rows),
            "max_completion_tokens":max((r.get("usage",{}).get("completion_tokens",0) for r in rows),default=0)}


def chat_overlap(chats):
    result = []
    for chat_index,chat in enumerate(chats):
        previous = set()
        for turn,row in enumerate(chat):
            words = re.findall(r"\w+",row["output"].lower())
            grams = Counter(tuple(words[i:i+8]) for i in range(max(0,len(words)-7)))
            overlapping = sum(count for gram,count in grams.items() if gram in previous)
            result.append({"chat":chat_index,"turn":turn,
                           "previous_answer_8gram_fraction":overlapping/max(1,sum(grams.values()))})
            previous.update(grams)
    return result


def main():
    result = {"scope":"Signals only: raw Bulgarian coherence, hostility, semantic repetition and turn relevance require review.",
              "conditions":{}}
    for path in sorted((OUT/"bulgarian").glob("*.json")):
        data = json.loads(path.read_text())
        rows = data["single_outputs"] + [r for chat in data["conversations"] for r in chat]
        result["conditions"][str(path.relative_to(OUT))] = {**summary(rows),
            "chat_overlap":chat_overlap(data["conversations"])}
    for path in sorted(OUT.glob("*confirmation/results.json")):
        data = json.loads(path.read_text())
        rows = data["candidate"] + [r for chat in data["conversations"] for r in chat]
        result["conditions"][str(path.relative_to(OUT))] = {**summary(rows),
            "unsteered":summary(data["unsteered"]),"chat_overlap":chat_overlap(data["conversations"])}
    for name in ("browser_with_tools.json","browser_final.json"):
        path = OUT/name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows = [{**r,**flags(r)} for r in data["outputs"]]
        chats = [[r for r in rows if r["kind"]=="multi_turn"]]
        result["conditions"][str(path.relative_to(OUT))] = {**summary(rows),"chat_overlap":chat_overlap(chats)}
    (OUT/"operational_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for name,row in result["conditions"].items():
        print(name,{k:v for k,v in row.items() if k not in ("chat_overlap",)})


if __name__ == "__main__":
    main()
