"""
Pull real GDC clinical data for the 90 non-BRCA pan-cancer training patients
(the 59 BRCA patients already have this in cohort_clinical.csv). Same fields,
same GDC cases API, so the two tables can be concatenated into one real
149-patient clinical table for pan-cancer EDA.
"""
import time
import requests
import pandas as pd

df = pd.read_csv("data/tcga_training/pancancer_file_to_submitter.csv")
FIELDS = [
    "submitter_id", "case_id", "demographic.sex_at_birth", "demographic.race",
    "demographic.ethnicity", "demographic.vital_status", "demographic.age_at_index",
    "diagnoses.primary_diagnosis", "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t", "diagnoses.ajcc_pathologic_n", "diagnoses.ajcc_pathologic_m",
    "diagnoses.morphology",
]

rows = []
for i, sub in enumerate(df["submitter_id"].unique()):
    r = requests.get(
        "https://api.gdc.cancer.gov/cases",
        params={
            "filters": '{"op":"=","content":{"field":"submitter_id","value":"%s"}}' % sub,
            "fields": ",".join(FIELDS),
            "format": "JSON",
            "size": "1",
        },
        timeout=30,
    )
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    if not hits:
        print("no hit for", sub)
        continue
    h = hits[0]
    demo = h.get("demographic", {})
    diag = (h.get("diagnoses") or [{}])[0]
    rows.append({
        "case_id": h.get("case_id"), "submitter_id": h.get("submitter_id"),
        "sex_at_birth": demo.get("sex_at_birth"), "race": demo.get("race"),
        "ethnicity": demo.get("ethnicity"), "vital_status": demo.get("vital_status"),
        "age_at_index": demo.get("age_at_index"),
        "primary_diagnosis": diag.get("primary_diagnosis"),
        "ajcc_pathologic_stage": diag.get("ajcc_pathologic_stage"),
        "ajcc_pathologic_t": diag.get("ajcc_pathologic_t"),
        "ajcc_pathologic_n": diag.get("ajcc_pathologic_n"),
        "ajcc_pathologic_m": diag.get("ajcc_pathologic_m"),
        "morphology": diag.get("morphology"),
    })
    if i % 15 == 0:
        print(f"{i+1}/{len(df['submitter_id'].unique())}")
    time.sleep(0.15)

out = pd.DataFrame(rows)
out = out.merge(df[["submitter_id", "cancer_type"]].drop_duplicates(), on="submitter_id", how="left")
out.to_csv("data/tcga_training/pancancer_clinical.csv", index=False)
print("wrote", len(out), "rows")
print(out["cancer_type"].value_counts())
