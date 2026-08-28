"""Download the batch of real TCGA-BRCA MAF files (excluding the patient file) used as somatic training labels."""
import json
import time
import requests

PATIENT_FILE_ID = "fb319d50-23ab-4c2b-9981-d296fdeeb983"

with open("data/tcga_training/file_ids.json") as f:
    hits = json.load(f)

hits = [h for h in hits if h["file_id"] != PATIENT_FILE_ID]
print("training files to download:", len(hits))

for i, h in enumerate(hits):
    out_path = f"data/tcga_training/{h['file_id']}.maf.gz"
    r = requests.get(f"https://api.gdc.cancer.gov/data/{h['file_id']}", timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    if i % 10 == 0:
        print(f"downloaded {i+1}/{len(hits)}")
    time.sleep(0.2)

print("done")
