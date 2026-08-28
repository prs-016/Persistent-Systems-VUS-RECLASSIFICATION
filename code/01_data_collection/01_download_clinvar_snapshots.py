import os
import subprocess

FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited"
OUTPUT_DIR = "data/stage2"

SNAPSHOTS = {
    "old": (
        "variant_summary_2019-06.txt.gz",
        f"{FTP_BASE}/archive/2019/variant_summary_2019-06.txt.gz",
    ),
    "current": (
        "variant_summary_current.txt.gz",
        f"{FTP_BASE}/variant_summary.txt.gz",
    ),
}

# GeneSymbol, ClinicalSignificance, LastEvaluated, ReviewStatus, NumberSubmitters, VariationID
COLUMNS_TO_KEEP = "5,7,9,25,26,31"


def download_and_slim():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, (filename, url) in SNAPSHOTS.items():
        gz_path = os.path.join(OUTPUT_DIR, filename)
        slim_path = os.path.join(OUTPUT_DIR, f"{name}_slim.tsv")

        if not os.path.exists(gz_path) or os.path.getsize(gz_path) == 0:
            print(f"downloading {name}: {url}")
            subprocess.run(["curl", "-s", "-m", "60", "-o", gz_path, url], check=True)

        if not os.path.exists(slim_path):
            print(f"slimming {name} down to the columns we need")
            with open(slim_path, "w") as out_file:
                zcat = subprocess.Popen(["zcat", gz_path], stdout=subprocess.PIPE)
                subprocess.run(["cut", f"-f{COLUMNS_TO_KEEP}"], stdin=zcat.stdout, stdout=out_file)
                zcat.wait()

        print(f"{name}: {gz_path} -> {slim_path}")


if __name__ == "__main__":
    download_and_slim()
    print("done, next up is 02_build_resolution_labels.py")
