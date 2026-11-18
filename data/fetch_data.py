# download the public prompt sets used by direction.py into data/
# AdvBench (llm-attacks, MIT), Alpaca no-input subset (tatsu-lab/alpaca, CC BY-NC 4.0), JailbreakBench JBB-Behaviors (MIT)
import json, os, random, urllib.request
from datasets import load_dataset
D = os.path.dirname(os.path.abspath(__file__))
urllib.request.urlretrieve("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv", f"{D}/advbench_harmful_behaviors.csv")
d = load_dataset("tatsu-lab/alpaca", split="train").filter(lambda r: r["input"] == "")
rows = [r["instruction"] for r in d]; random.Random(0).shuffle(rows)
json.dump(rows[:2000], open(f"{D}/alpaca_noinput_2000.json", "w"), indent=0)
h = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
json.dump([{"goal": r["Goal"], "source": r["Source"], "category": r["Category"]} for r in h], open(f"{D}/jbb_harmful_meta.json", "w"), indent=0)
b = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
json.dump([r["Goal"] for r in b], open(f"{D}/jbb_benign.json", "w"), indent=0)
print("data ready")
