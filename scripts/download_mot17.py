"""Download MOT17 and unpack the labelled training sequences. Never commit the data.

MOT17 is CC BY-NC-SA 3.0 (non-commercial): https://motchallenge.net/data/MOT17/
Only the FRCNN copy of each training sequence is unpacked: the three detector folders
(DPM, FRCNN, SDP) hold the same images and the same ground truth.

    python scripts/download_mot17.py --dir D:/Datasets/MOT17
"""
import argparse
import pathlib
import urllib.request
import zipfile

URL = "https://motchallenge.net/data/MOT17.zip"


def download(url, dest):
    part = dest.with_suffix(".part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as r, open(part, "ab" if r.status == 206 else "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.rename(dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="where to store MOT17 (outside the repo)")
    out = pathlib.Path(p.parse_args().dir)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "MOT17.zip"
    if not zip_path.exists():
        print(f"downloading {URL} (about 5.9 GB)")
        download(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if "/train/" in "/" + n and "-FRCNN" in n]
        z.extractall(out, names)
    print(f"unpacked {len(names)} files to {out}")


if __name__ == "__main__":
    main()
