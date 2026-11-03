"""Build the Hugging Face Space folder for the Gradio demo, and upload it.

  python scripts/deploy_space.py --dry-run            # only build runs/space_staging and list it
  python scripts/deploy_space.py                      # build, then create or update the Space

Needs `hf auth login` once (a token that can write to Spaces) and huggingface_hub (installed with Gradio).
The Space gets the demo app, its requirements, the two example images and a card. It never gets raw
example footage: only the annotated screenshot and GIF made by this project.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARD = """---
title: track-counter
emoji: 🚶
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Count people crossing a line in a short video
---

# track-counter demo

Upload a short video, click two points to draw a counting line, and get the annotated video, the in and
out counts and `events.csv`. YOLOX-s (ONNX) detects, ByteTrack follows, and a line-crossing rule counts.
It runs on a free CPU, so a 10 second clip takes a while.

The counts show what the tool does, not how accurate it is. Measured accuracy, limits and the code are in
the repository: https://github.com/Onkar-Gupta/object-tracking-counter

The detector model is the official YOLOX-s ONNX file (Apache-2.0), downloaded on first use. Example
screenshots come from stock footage credited in the repository (docs/demo_credit.md).
"""


def build(out: Path):
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    shutil.copy(ROOT / "demo" / "app.py", out / "app.py")
    shutil.copy(ROOT / "demo" / "requirements.txt", out / "requirements.txt")
    shutil.copy(ROOT / "demo" / "packages.txt", out / "packages.txt")
    shutil.copy(ROOT / "LICENSE", out / "LICENSE")
    for name in ("gradio_demo.png", "demo.gif"):
        shutil.copy(ROOT / "docs" / name, out / "assets" / name)
    (out / "README.md").write_text(CARD, encoding="utf-8")
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="Onkar-Gupta-HF/track-counter")
    ap.add_argument("--staging", default=str(ROOT / "runs" / "space_staging"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    files = build(Path(args.staging))
    print("Space folder:", args.staging)
    for f in files:
        print("  ", f, f"({(Path(args.staging) / f).stat().st_size // 1024} KB)")
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="space", folder_path=args.staging,
                      commit_message="Deploy track-counter demo")
    print(f"Uploaded. Build log and app: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
