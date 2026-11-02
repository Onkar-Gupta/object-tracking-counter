"""Counter and tracker ablation, as set out in eval/ablation_protocol.md (committed before running).

  python scripts/ablate_counter.py --mot-dir D:/Datasets/MOT17/MOT17/train \
      --detector "public FRCNN=D:/Datasets/MOT17/MOT17/train" \
      --detector "YOLOX-s tiled=D:/Datasets/detections/yolox_s_tiled" --out reports/ablation.json
"""
import argparse
import itertools
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from track_counter.evaluate import evaluate_sequences, paired_recall_difference

N_FRAMES, MARGINS, MIN_CONSEC = (3, 5, 8, 12), (4, 8, 12), (2, 4)
TUNING = ("MOT17-02-FRCNN", "MOT17-04-FRCNN")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mot-dir", required=True)
    ap.add_argument("--detector", action="append", required=True, help='"name=detections dir"')
    ap.add_argument("--exploratory", default="reports/exploratory_counting.json",
                    help="where each detector's score setting t was chosen")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--lines", default="eval/lines.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ev, lines = cfg["evaluation"], yaml.safe_load(Path(args.lines).read_text())
    base_counter = {k: v for k, v in cfg["counter"].items() if k != "line"}
    chosen_t = {n: d["chosen_t"] for n, d in json.loads(Path(args.exploratory).read_text())["detectors"].items()}
    tuning_lines = {"sequences": {s: lines["sequences"][s] for s in TUNING}}

    def run(det_dir, t, n, margin, mc, line_cfg, context):
        counter = {**base_counter, "min_frames_on_far_side": n, "margin_px": margin}
        tracker = {"high_conf_det_threshold": t, "track_activation_threshold": t,
                   "minimum_consecutive_frames": mc}
        return evaluate_sequences(args.mot_dir, line_cfg, "runs/latest/mot", ev["min_conf"], counter,
                                  tracker, ev["tolerance_s"], ev["position_tolerance"],
                                  ev["gt_margin_px"], det_dir, context)

    out, reports = {"protocol": "eval/ablation_protocol.md", "detectors": {}}, {}
    for spec in args.detector:
        name, det_dir = spec.split("=", 1)
        t, trials = chosen_t[name], []
        for n, margin, mc in tqdm(list(itertools.product(N_FRAMES, MARGINS, MIN_CONSEC)),
                                  desc=f"{name} (t={t})", unit="setting"):
            p = run(det_dir, t, n, margin, mc, tuning_lines, False)["pooled_over_all_lines"]
            trials.append({"n": n, "margin_px": margin, "min_consecutive": mc, "precision": p["precision"],
                           "recall": p["recall"], "f1": p["f1"], "pred_events": p["pred_events"]})
        best = max(trials, key=lambda r: (round(r["f1"], 9), r["n"], r["margin_px"]))
        reports[name] = run(det_dir, t, best["n"], best["margin_px"], best["min_consecutive"], lines, True)
        out["detectors"][name] = {"t": t, "grid_on_02_04": trials, "chosen": best, "result": reports[name]}
    names = list(reports)
    for other in names[1:]:
        d = paired_recall_difference(reports[other], reports[names[0]])
        prec = reports[other]["pooled_over_all_lines"]["precision"]
        out["detectors"][other]["vs_baseline"] = {
            "baseline": names[0], "recall_difference": d, "precision": prec,
            "recall_gain_excludes_zero": bool(d["ci"] and d["ci"][0] > 0),
            "precision_at_least_0.80": prec >= 0.80}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n{'detector':16s} chosen (N, margin, minconsec) | all three: P     R     F1   events | on 09 only: P     R")
    for n in names:
        r, c = out["detectors"][n]["result"], out["detectors"][n]["chosen"]
        p, s9 = r["pooled_over_all_lines"], r["sequences"]["MOT17-09-FRCNN"]["lines"]
        gt9, tp9, pr9 = (sum(v[k] for v in s9.values()) for k in ("gt_events", "tp", "pred_events"))
        print(f"{n:16s} ({c['n']:2d}, {c['margin_px']:2d}, {c['min_consecutive']})              | "
              f"{p['precision']:.3f} {p['recall']:.3f} {p['f1']:.3f} {p['pred_events']:4d}   | "
              f"{tp9 / pr9 if pr9 else 0:.3f} {tp9 / gt9:.3f}")
        vb = out["detectors"][n].get("vs_baseline")
        if vb:
            d = vb["recall_difference"]
            print(f"{'':16s} vs tuned {vb['baseline']}: recall {d['difference']:+.3f} CI [{d['ci'][0]:+.3f}, "
                  f"{d['ci'][1]:+.3f}], precision {vb['precision']:.3f} -> "
                  f"{'GAIN' if vb['recall_gain_excludes_zero'] and vb['precision_at_least_0.80'] else 'no gain claimed'}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
