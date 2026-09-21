"""Capture the empirical stream of learning events from a real series.

Careers cannot be simulated by running 60,000 real matches, but a career
model is only trustworthy if it is driven by the situation and outcome
statistics the engine ACTUALLY produces. So: record every (situation,
action, valence, significance) the match loop generates, then replay that
empirical distribution at career scale.
"""
import sys, collections, statistics, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import instincts
from series import SeriesRunner

records = []
_real_learn = instincts.InstinctBank.learn


def spy(self, situation, action, valence, significance):
    records.append({
        "situation": list(situation.as_tuple()),
        "action": action,
        "valence": valence,
        "significance": significance,
    })
    return _real_learn(self, situation, action, valence, significance)


instincts.InstinctBank.learn = spy

runner = SeriesRunner(matches=10, seed=42).play()
instincts.InstinctBank.learn = _real_learn

print(f"learning events over 10 matches, 22 players: {len(records)}")
print(f"  per player per match: {len(records)/22/10:.1f}")
print()

actions = collections.Counter(r["action"] for r in records)
print("action mix:")
for a, n in actions.most_common():
    print(f"   {a:<14}{n:>6}  {n/len(records):>6.1%}")
print()

pos = [r for r in records if r["valence"] > 0]
print(f"valence: {len(pos)/len(records):.1%} positive (anchors), "
      f"{1-len(pos)/len(records):.1%} negative (traumas)")
sig = [r["significance"] for r in records]
print(f"significance: mean {statistics.mean(sig):.2f}  "
      f"min {min(sig):.2f}  max {max(sig):.2f}")
print()

print("situation dimension distributions (the space careers are lived in):")
names = ["pressure", "progression", "time_crit", "density", "support", "width", "load"]
for i, name in enumerate(names):
    vals = [r["situation"][i] for r in records]
    print(f"   {name:<12} mean {statistics.mean(vals):.2f}  "
          f"sd {statistics.pstdev(vals):.2f}  "
          f"range {min(vals):.2f}-{max(vals):.2f}")

with open(pathlib.Path(__file__).parent / 'stream.json', 'w') as fh:
    json.dump(records, fh)
print(f"\nwrote {len(records)} records to stream.json")
