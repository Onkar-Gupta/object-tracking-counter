"""Step C1 ablation: tracker choice and anchor smoothing, as set out in eval/tracker_protocol.md
(committed before running).

  python scripts/ablate_trackers.py --mot-dir D:/Datasets/MOT17/MOT17/train \\
      --detector "public FRCNN=D:/Datasets/MOT17/MOT17/train" \\
      --detector "YOLOX-s tiled=D:/Datasets/detections/yolox_s_tiled" --out reports/tracker_ablation.json
"""
import argparse
import itertools
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from track_counter.evaluate import evaluate_sequences, paired_recall_difference

ANCHORS = ("bottom_center", "center")
TRACKERS, ALPHAS, N_FRAMES = ("bytetrack", "ocsort", "botsort"), (1.0, 0.5, 0.3), (5, 12, 20, 30)
MARGIN = 4.0
TUNING = ("MOT17-02-FRCNN", "MOT17-04-FRCNN")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mot-dir", required=True)
    ap.add_argument("--detector", action="append", required=True, help='"name=detections dir"')
    ap.add_argument("--ablation", default="reports/ablation.json", help="score and tracker settings")
    ap.add_argument("--cleanup", default="reports/cleanup_ablation.json", help="clean-up settings")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--lines", default="eval/lines.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ev, lines = cfg["evaluation"], yaml.safe_load(Path(args.lines).read_text())
    base = {k: v for k, v in cfg["counter"].items() if k != "line"}
    abl = json.loads(Path(args.ablation).read_text())["detectors"]
    clean = json.loads(Path(args.cleanup).read_text())["anchors"]
    tuning_lines = {"sequences": {s: lines["sequences"][s] for s in TUNING}}

    def run(det_dir, name, anchor, tracker, alpha, n, line_cfg, context):
        c = clean[anchor][name]["chosen"]  # clean-up settings chosen for this detector and anchor
        counter = {**base, "anchor": anchor, "min_frames_on_far_side": n, "margin_px": MARGIN,
                   "cooldown_frames": c["cooldown_frames"], "merge_frames": c["merge_frames"],
                   "merge_px": c["merge_px"], "min_track_frames": c["min_track_frames"],
                   "smooth_alpha": alpha}
        t = abl[name]["t"]
        params = {"name": tracker, "high_conf_det_threshold": t,
                  "minimum_consecutive_frames": abl[name]["chosen"]["min_consecutive"]}
        if tracker != "ocsort":  # OC-SORT has no activation threshold
            params["track_activation_threshold"] = t
        return evaluate_sequences(args.mot_dir, line_cfg, "runs/latest/mot", ev["min_conf"], counter,
                                  params, ev["tolerance_s"], ev["position_tolerance"],
                                  ev["gt_margin_px"], det_dir, context)

    order = {t: i for i, t in enumerate(TRACKERS)}
    out = {"protocol": "eval/tracker_protocol.md", "anchors": {}}
    for anchor in ANCHORS:
        reports, entry = {}, {}
        for spec in args.detector:
            name, det_dir = spec.split("=", 1)
            trials = []
            for tracker, alpha, n in tqdm(list(itertools.product(TRACKERS, ALPHAS, N_FRAMES)),
                                          desc=f"{anchor} / {name}", unit="setting"):
                p = run(det_dir, name, anchor, tracker, alpha, n, tuning_lines, False)["pooled_over_all_lines"]
                trials.append({"tracker": tracker, "smooth_alpha": alpha, "n": n,
                               "precision": p["precision"], "recall": p["recall"], "f1": p["f1"],
                               "pred_events": p["pred_events"], "gt_events": p["gt_events"]})
            best = max(trials, key=lambda r: (round(r["f1"], 9), r["n"], -r["smooth_alpha"],
                                              -order[r["tracker"]]))
            reports[name] = run(det_dir, name, anchor, best["tracker"], best["smooth_alpha"], best["n"],
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
        print(f"{'detector':16s} chosen (tracker, alpha, N)  | all three: P     R     F1   events/GT | on 09: P     R")
        for n, d in entry.items():
            r, c = d["result"], d["chosen"]
            p, s9 = r["pooled_over_all_lines"], r["sequences"]["MOT17-09-FRCNN"]["lines"]
            gt9, tp9, pr9 = (sum(v[k] for v in s9.values()) for k in ("gt_events", "tp", "pred_events"))
            print(f"{n:16s} ({c['tracker']:9s}, {c['smooth_alpha']:.1f}, {c['n']:2d}) | {p['precision']:.3f} "
                  f"{p['recall']:.3f} {p['f1']:.3f} {p['pred_events']:4d}/{p['gt_events']:3d}  | "
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
