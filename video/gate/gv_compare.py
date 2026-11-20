"""Table of Charades-STA clip-scan pilot analyses (scan/pilot_analyze.py outputs), one column per (file, scorer).

python3 video/gate/gv_compare.py v1=outputs/scan/pilot_clips_short_analysis.json:g_yes gvlast=...:g_yes gvfinal=...:g_yes
"""
import json, sys

cols = []
for arg in sys.argv[1:]:
    name, rest = arg.split("=", 1)
    path, scorer = rest.rsplit(":", 1)
    cols.append((name, json.load(open(path))["scorers"][scorer]))


def g(d, *ks):
    for k in ks:
        d = d[k]
    return d


rows = [("clip AUROC", lambda d: d["frame_auroc"]), ("scan-max AUROC", lambda d: d["scan_max_auroc"])]
for N in ("8", "32", "128"):
    rows += [(f"naive set FA N={N}", lambda d, N=N: g(d, "none_of_these", N, "naive", "false_alarm")),
             (f"maxstat set FA N={N}", lambda d, N=N: g(d, "none_of_these", N, "maxstat", "false_alarm")),
             (f"maxstat miss N={N}", lambda d, N=N: g(d, "none_of_these", N, "maxstat", "miss"))]
rows += [("first-occ. early/FA", lambda d: d["first_occurrence"]["early_or_false_alarm"]),
         ("first-occ. detect", lambda d: d["first_occurrence"]["detect_rate"]),
         ("first-occ. median frames after onset", lambda d: d["first_occurrence"]["median_frames_after_onset"])]
print("| metric | " + " | ".join(n for n, _ in cols) + " |")
print("|---|" + "---|" * len(cols))
for label, f in rows:
    vals = []
    for _, d in cols:
        try:
            vals.append(f"{f(d):.3f}")
        except (KeyError, TypeError):
            vals.append("")
    print(f"| {label} | " + " | ".join(vals) + " |")
