"""
Download RWF-2000 dataset from available sources.
Tries multiple mirrors in order.
"""
import sys, os, zipfile, shutil, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(r"c:\deep_lr_project")
DEST = ROOT / "datasets" / "rwf2000"

def download_file(url, dest_path, label=""):
    """Download with progress bar."""
    print(f"Downloading {label} from:\n  {url}")
    
    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            bar = '#' * (pct // 2) + '-' * (50 - pct // 2)
            print(f"\r  [{bar}] {pct}%", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress)
        print()
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


# Check if already downloaded
existing = list(DEST.rglob("*.avi")) + list(DEST.rglob("*.mp4"))
if len(existing) > 100:
    print(f"RWF-2000 already present: {len(existing)} files found.")
    sys.exit(0)

print("RWF-2000 not found. Attempting download...")
print("Note: RWF-2000 is a 4.7GB dataset. Checking available sources...\n")

# Try Kaggle API first
try:
    import kaggle
    print("Kaggle available. Trying kaggle datasets download...")
    # kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", "mohamedmustafa/real-life-violence-situations-dataset",
        "-p", str(ROOT / "datasets" / "tmp"),
        "--unzip"
    ], capture_output=True, text=True, timeout=3600)
    if result.returncode == 0:
        print("Kaggle download successful!")
        sys.exit(0)
    else:
        print(f"Kaggle error: {result.stderr[:200]}")
except ImportError:
    pass
except Exception as e:
    print(f"Kaggle attempt failed: {e}")

# Try direct GitHub release (sample subset only - the full dataset requires auth)
# RWF-2000 official repo does not host raw videos publicly
# Hockey Fight dataset IS publicly available on Kaggle
print("\nRWF-2000 requires authentication (GitHub/Kaggle account).")
print("Attempting Hockey Fight Detection dataset instead (public)...")

# Hockey dataset on Kaggle (public, no auth needed for direct URL)
# Try to get a small publicly available violence dataset for bootstrap training
urls_to_try = [
    # Hockey Fight from public archive
    ("https://github.com/liuguanfu/Violence-Detection/raw/refs/heads/main/Hockey.zip",
     "Hockey Fight Dataset"),
]

DEST_HOCKEY = ROOT / "datasets" / "hockey"
DEST_HOCKEY.mkdir(parents=True, exist_ok=True)
tmp = ROOT / "datasets" / "tmp_download.zip"

for url, label in urls_to_try:
    if download_file(url, str(tmp), label):
        try:
            print(f"Extracting to {DEST_HOCKEY}...")
            with zipfile.ZipFile(str(tmp), 'r') as z:
                z.extractall(str(DEST_HOCKEY.parent))
            tmp.unlink(missing_ok=True)
            files = list(DEST_HOCKEY.rglob("*.avi")) + list(DEST_HOCKEY.rglob("*.mp4"))
            print(f"Extracted {len(files)} video files.")
            if files:
                sys.exit(0)
        except Exception as e:
            print(f"Extraction failed: {e}")

print("\nAutomatic download unsuccessful.")
print("\nMANUAL DOWNLOAD INSTRUCTIONS:")
print("="*55)
print("Option 1 – RWF-2000 (recommended, 4.7GB):")
print("  https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection")
print("  → Place in: datasets/rwf2000/train/Fight/ and datasets/rwf2000/train/NonFight/")
print()
print("Option 2 – Hockey Fight (simpler, 1GB):")
print("  https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes")
print("  → Place in: datasets/hockey/Fight/ and datasets/hockey/NonFight/")
print()
print("After placing files, run:  python train_model.py --dataset rwf2000")
sys.exit(1)
