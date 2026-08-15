"""Download the YCB assets this experiment needs.

    python scripts/download_ycb.py --config configs/default.yaml

Fetches, for each of the five objects, the google_16k mesh (used to render the
synthetic training data) and the RGB photographs (used as the held-out real
test set). Roughly 2-3 GB total; already-downloaded objects are skipped.

The YCB object and model set is published by Calli et al. and carries its own
license terms - see https://www.ycbbenchmarks.com/
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import load_config  # noqa: E402

BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data"


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"    cached: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        print(f"    fetching {url.rsplit('/', 1)[-1]} ...", end="", flush=True)
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dest)
        print(" ok")
        return True
    except Exception as e:
        print(f" FAILED ({e})")
        tmp.unlink(missing_ok=True)
        return False


def extract(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)


def main():
    ap = argparse.ArgumentParser(description="Download YCB meshes and photographs")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache = cfg.ycb_meshes.parent / "_archives"

    print("YCB assets")
    print(f"  meshes -> {cfg.ycb_meshes}")
    print(f"  photos -> {cfg.real_test}\n")

    failed = []
    for obj in cfg.objects:
        print(f"  {obj}")

        if cfg.mesh_path(obj).exists():
            print("    mesh present - skipping")
        else:
            arc = cache / f"{obj}_google_16k.tgz"
            if download(f"{BASE_URL}/google/{obj}_google_16k.tgz", arc):
                extract(arc, cfg.ycb_meshes)
            else:
                failed.append(f"{obj} (mesh)")

        if (cfg.real_test / obj).exists():
            print("    photographs present - skipping")
        else:
            arc = cache / f"{obj}_berkeley_rgb_highres.tgz"
            if download(f"{BASE_URL}/berkeley/{obj}/{obj}_berkeley_rgb_highres.tgz", arc):
                extract(arc, cfg.real_test)
            else:
                failed.append(f"{obj} (photographs)")

    if not args.keep_archives and cache.exists():
        shutil.rmtree(cache, ignore_errors=True)

    print("\nVerification")
    ok = True
    for obj in cfg.objects:
        mesh = "OK" if cfg.mesh_path(obj).exists() else "MISSING"
        photos = cfg.real_test / obj
        n = 0
        if photos.exists():
            for pattern in ("*.jpg", "*.JPG", "*.jpeg"):
                n += len([p for p in photos.rglob(pattern)
                          if "masks" not in p.parts and "poses" not in p.parts])
        print(f"  {obj:<24} mesh {mesh:<8} photographs {n}")
        ok = ok and mesh == "OK" and n > 0

    if failed:
        print("\nFailed:")
        for f in failed:
            print(f"  {f}")
        print("\nThe YCB S3 endpoint moves occasionally. If downloads fail, grab "
              "the archives manually from https://www.ycbbenchmarks.com/ and "
              "extract them into the two paths above.")
        sys.exit(1)

    print("\nAll assets ready." if ok else "\nSome assets missing - see above.")


if __name__ == "__main__":
    main()