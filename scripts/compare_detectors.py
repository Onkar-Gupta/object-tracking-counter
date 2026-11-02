"""Exploratory stage 2: does a different detector improve counting? Same lines, same ground truth.

Each detector gets one score setting t, chosen on MOT17-02 and MOT17-04 only by pooled event F1.
t sets the tracker's high-confidence split and track-activation threshold (they depend on the
detector's score scale), with a fixed score floor of 0.1. The chosen t is then run on all three
sequences. Every detector, including the public FRCNN detections, is tuned the same way.
The first --detector is the baseline; the others are compared with it by a paired person-level
bootstrap on event recall.

  python scripts/compare_detectors.py --mot-dir D:/Datasets/MOT17/MOT17/train \\
      --detector "public FRCNN=D:/Datasets/MOT17/MOT17/train" \\
      --detector "YOLOX-s tiled=D:/Datasets/detections/yolox_s_tiled" --out reports/exploratory_counting.json
"""
import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from track_counter.evaluate import evaluate_sequences, paired_recall_difference

GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
TUNING = ("MOT17-02-FRCNN", "MOT17-04-FRCNN")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mot-dir", required=True)
    ap.add_argument("--detector", action="append", required=True, help='"name=detections dir"')
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--lines", default="eval/lines.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ev, lines = cfg["evaluation"], yaml.safe_load(Path(args.lines).read_text())
    counter = {k: v for k, v in cfg["counter"].items() if k != "line"}
    tuning_lines = {"sequences": {s: lines["sequences"][s] for s in TUNING}}

    def run(det_dir, t, line_cfg, context):
        return evaluate_sequences(
            args.mot_dir, line_cfg, "runs/latest/mot", ev["min_conf"], counter,
            {"high_conf_det_threshold": t, "track_activation_threshold": t},
            ev["tolerance_s"], ev["position_tolerance"], ev["gt_margin_px"], det_dir, context)

    out, reports = {"grid": list(GRID), "tuning_sequences": list(TUNING), "detectors": {}}, {}
    for spec in args.detector:
        name, det_dir = spec.split("=", 1)
        table = []
        for t in tqdm(GRID, desc=f"{name}: choosing score setting on 02+04", unit="setting"):
            p = run(det_dir, t, tuning_lines, False)["pooled_over_all_lines"]
            table.append({"t": t, "precision": p["precision"], "recall": p["recall"], "f1": p["f1"]})
        best = max(table, key=lambda r: r["f1"])
        print(f"{name}: chose t={best['t']} (F1 {best['f1']:.3f} on 02+04). Running all three sequences...",
              flush=True)
        reports[name] = run(det_dir, best["t"], lines, True)
        out["detectors"][name] = {"selection_on_02_04": table, "chosen_t": best["t"],
                                  "result": reports[name]}
    names = list(reports)
    base = names[0]
    for other in names[1:]:
        d = paired_recall_difference(reports[other], reports[base])
        prec = reports[other]["pooled_over_all_lines"]["precision"]
        out["detectors"][other]["vs_baseline"] = {
            "baseline": base, "recall_difference": d, "precision": prec,
            "recall_gain_excludes_zero": bool(d["ci"] and d["ci"][0] > 0),
            "precision_at_least_0.80": prec >= 0.80}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n{'detector':16s} t    | P     R     F1    | recall on 09 only | vs {base}")
    for n in names:
        r = out["detectors"][n]["result"]
        p, s9 = r["pooled_over_all_lines"], r["sequences"]["MOT17-09-FRCNN"]["lines"]
        gt9 = sum(v["gt_events"] for v in s9.values())
        tp9 = sum(v["tp"] for v in s9.values())
        vb = out["detectors"][n].get("vs_baseline")
        extra = "" if not vb else (f"recall diff {vb['recall_difference']['difference']:+.3f} "
                                   f"CI [{vb['recall_difference']['ci'][0]:+.3f}, {vb['recall_difference']['ci'][1]:+.3f}]")
        print(f"{n:16s} {out['detectors'][n]['chosen_t']:.1f}  | {p['precision']:.3f} {p['recall']:.3f} "
              f"{p['f1']:.3f} | {tp9 / gt9:.3f}             | {extra}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
