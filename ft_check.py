"""Diagnose the free-transfer estimate week by week.

Usage (from the folder that contains requirements.txt):
    python ft_check.py <team-id>

Hits give hard evidence: in a week where you paid -4s, the free transfers you
had that week = transfers made - hit_cost/4. Any week where the replay disagrees
with that evidence is flagged.
"""
import sys

from fpl_agent.config import MAX_FREE_TRANSFERS
from fpl_agent.data import FPLClient


def report(entry: dict, hist: dict) -> list[str]:
    started = int(entry.get("started_event") or 1)
    cur = int(entry.get("current_event") or 0)
    chips = hist.get("chips", [])
    chip_at = {c["event"]: c["name"] for c in chips}
    by_gw = {h["event"]: h for h in hist.get("current", [])}
    out = [f"started_event={started}  current_event={cur}",
           "chips as FPL names them: " + (", ".join(f"GW{c['event']}={c['name']}" for c in chips) or "none"),
           "",
           "GW   FT(replay)  made  hit_cost  chip       note"]
    ft = 1
    for g in range(started + 1, cur + 1):
        h = by_gw.get(g, {})
        made, cost = h.get("event_transfers", 0), h.get("event_transfers_cost", 0)
        chip = chip_at.get(g, "")
        note = ""
        if cost:
            actual = made - cost // 4
            note = f"hits prove FT was {actual}" + ("" if actual == ft else "  <-- MISMATCH")
        elif not chip and made > ft:
            note = f"made {made} with no hit, so FT was >= {made}  <-- replay too LOW"
        out.append(f"{g:<4} {ft:<11} {made:<5} {cost:<9} {chip:<10} {note}")
        if chip in ("wildcard", "freehit"):
            continue
        ft = min(MAX_FREE_TRANSFERS, max(0, ft - made) + 1)
    out += ["", f"Replayed free transfers for GW{cur + 1}: {ft}"]
    return out


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        sys.exit("usage: python ft_check.py <team-id>")
    eid = int(sys.argv[1])
    client = FPLClient(cache_dir=".fpl_cache")
    entry, hist = client.get(f"entry/{eid}"), client.get(f"entry/{eid}/history")
    print("\n".join(report(entry, hist)))


if __name__ == "__main__":
    main()
