"""Step B counter clean-up ablation, as set out in eval/cleanup_protocol.md (committed before running).

  python scripts/ablate_cleanup.py --mot-dir D:/Datasets/MOT17/MOT17/train \\
      --detector "public FRCNN=D:/Datasets/MOT17/MOT17/train" \\
      --detector "YOLOX-s tiled=D:/Datasets/detections/yolox_s_tiled" --out reports/cleanup_ablation.json
"""
import argparse
import itertools
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from track_counter.evaluate import evaluate_sequences, paired_recall_difference

ANCHORS = ("bottom_center", "center")
N_FRAMES, COOLDOWNS, MERGES, MIN_AGES = (5, 12, 20, 30), (0, 30), ((0, 0), (15, 60)), (0, 10)
MARGIN = 4.0
TUNING = ("MOT17-02-FRCNN", "MOT17-04-FRCNN")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mot-dir", required=True)
    ap.add_argument("--detector", action="append", required=True, help='"name=detections dir"')
    ap.add_argument("--ablation", default="reports/ablation.json",
                    help="where each detector's score setting and tracker setting were chosen")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--lines", default="eval/lines.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ev, lines = cfg["evaluation"], yaml.safe_load(Path(args.lines).read_text())
    base = {k: v for k, v in cfg["counter"].items() if k != "line"}
    abl = json.loads(Path(args.ablation).read_text())["detectors"]
    tuning_lines = {"sequences": {s: lines["sequences"][s] for s in TUNING}}

    def run(det_dir, name, anchor, n, cooldown, merge, min_age, line_cfg, context):
        counter = {**base, "anchor": anchor, "min_frames_on_far_side": n, "margin_px": MARGIN,
                   "cooldown_frames": cooldown, "merge_frames": merge[0], "merge_px": merge[1],
                   "min_track_frames": min_age}
        t = abl[name]["t"]
        tracker = {"high_conf_det_threshold": t, "track_activation_threshold": t,
                   "minimum_consecutive_frames": abl[name]["chosen"]["min_consecutive"]}
        return evaluate_sequences(args.mot_dir, line_cfg, "runs/latest/mot", ev["min_conf"], counter,
                                  tracker, ev["tolerance_s"], ev["position_tolerance"],
                                  ev["gt_margin_px"], det_dir, context)

    out = {"protocol": "eval/cleanup_protocol.md", "anchors": {}}
    for anchor in ANCHORS:
        reports, entry = {}, {}
        for spec in args.detector:
            name, det_dir = spec.split("=", 1)
            trials = []
            for n, cd, merge, age in tqdm(list(itertools.product(N_FRAMES, COOLDOWNS, MERGES, MIN_AGES)),
                                          desc=f"{anchor} / {name}", unit="setting"):
                p = run(det_dir, name, anchor, n, cd, merge, age, tuning_lines, False)["pooled_over_all_lines"]
                trials.append({"n": n, "cooldown_frames": cd, "merge_frames": merge[0], "merge_px": merge[1],
                               "min_track_frames": age, "precision": p["precision"], "recall": p["recall"],
                               "f1": p["f1"], "pred_events": p["pred_events"], "gt_events": p["gt_events"]})
            best = max(trials, key=lambda r: (round(r["f1"], 9), r["n"], r["cooldown_frames"],
                                              r["merge_frames"] > 0, r["min_track_frames"]))
            reports[name] = run(det_dir, name, anchor, best["n"], best["cooldown_frames"],
                                (best["merge_frames"], best["merge_px"]), best["min_track_frames"],
                                lines, True)
            entry[name] = {"grid_on_02_04": trials, "chosen": best, "result": reports[name]}
        names = list(reports)
        for other in names[1:]:
            d = paired_recall_difference(reports[other], reports[names[0]])
            prec = reports[other]["pooled_over_all_lines"]["precision"]
            entry[other]["vs_baseline"] = {
                "baseline": names[0], "recall_difference": d, "precision": prec,
                "recall_gain_excludes_zero": bool(d["ci"] and d["ci"][0] > 0),
                "precision_at_least_0.80": prec >= 0.80}
        out["anchors"][anchor] = entry
    Path(args.out).write_text(json.dumps(out, indent=2))
    for anchor, entry in out["anchors"].items():
        print(f"\n=== anchor: {anchor} ===")
        print(f"{'detector':16s} chosen (N, cooldown, merge, min age) | all three: P     R     F1   events/GT | on 09: P     R")
        for n, d in entry.items():
            r, c = d["result"], d["chosen"]
            p, s9 = r["pooled_over_all_lines"], r["sequences"]["MOT17-09-FRCNN"]["lines"]
            gt9, tp9, pr9 = (sum(v[k] for v in s9.values()) for k in ("gt_events", "tp", "pred_events"))
            print(f"{n:16s} ({c['n']:2d}, {c['cooldown_frames']:2d}, {'on ' if c['merge_frames'] else 'off'}, "
                  f"{c['min_track_frames']:2d})                 | {p['precision']:.3f} {p['recall']:.3f} "
                  f"{p['f1']:.3f} {p['pred_events']:4d}/{p['gt_events']:3d}  | "
                  f"{tp9 / pr9 if pr9 else 0:.3f} {tp9 / gt9:.3f}")
            vb = d.get("vs_baseline")
            if vb:
                x = vb["recall_difference"]
                print(f"{'':16s} vs tuned {vb['baseline']}: recall {x['difference']:+.3f} CI [{x['ci'][0]:+.3f}, "
                      f"{x['ci'][1]:+.3f}], precision {vb['precision']:.3f} -> "
                      f"{'GAIN' if vb['recall_gain_excludes_zero'] and vb['precision_at_least_0.80'] else 'no gain claimed'}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
