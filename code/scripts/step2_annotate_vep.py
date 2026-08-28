"""
Step 2 — annotate the patient's variants against ClinVar, gnomAD, COSMIC, dbSNP.

Decision (Step 0): a local VEP install with custom ClinVar/gnomAD/COSMIC
source VCFs is infeasible in this environment (gnomAD's full VCF is hundreds
of GB, no VEP binary/cache present, no COSMIC license). Using the Ensembl
VEP REST API instead — one batch POST to /vep/human/region returns
ClinVar clin_sig, gnomAD allele frequencies, COSMIC IDs, and dbSNP rsIDs in
a single pass, verified against real Ensembl responses before writing this.

ANNOVAR addition (this session, per explicit request): a real local ANNOVAR
install now lives at stage1/annovar/ (annovar.latest.tar.gz downloaded via
the registration-gated link from openbioinformatics.org, perl scripts +
hg38 humandb files pulled for real). It runs as a second, independent
annotation source alongside the VEP REST call above — not a replacement.
Databases installed: refGeneWithVer (gene/consequence), clinvar_20221231
(ClinVar CLNSIG), gnomad211_exome (gnomAD exome AF). COSMIC is NOT included
via ANNOVAR — same reason as the VEP path: ANNOVAR's COSMIC tables require
a separate COSMIC license/login not available in this environment, so
COSMIC coverage still comes only from VEP's colocated_variants. dbSNP
(avsnp151) was evaluated but skipped: the ANNOVAR-formatted file is ~8 GB,
and dbSNP rsIDs are already covered by the VEP REST path, so downloading it
would only duplicate data already in the pipeline for a large disk cost.
ANNOVAR output columns are merged in with an `annovar_` prefix so they're
distinguishable from the VEP-sourced columns already in this file.
"""
import os
import subprocess
import tempfile
import time
import requests
import pandas as pd

VEP_REST_URL = "https://rest.ensembl.org/vep/human/region"
BATCH_SIZE = 200  # Ensembl VEP REST POST batch limit

# The public REST endpoint occasionally returns 503 (overloaded) or 429
# (rate limited), both transient. A couple of retries with backoff clears
# most of these without failing the whole run over a momentary blip.
VEP_RETRYABLE_STATUS = {429, 503}
VEP_MAX_ATTEMPTS = 4
VEP_RETRY_BASE_DELAY = 2.0  # seconds; doubles each retry

ANNOVAR_DIR = os.path.join(os.path.dirname(__file__), "..", "annovar")
ANNOVAR_HUMANDB = os.path.join(ANNOVAR_DIR, "humandb")
ANNOVAR_BUILD = "hg38"
ANNOVAR_PROTOCOL = "refGeneWithVer,clinvar_20221231,gnomad211_exome"
ANNOVAR_OPERATION = "g,f,f"


def build_region_string(row) -> str:
    chrom = row["chrom"].replace("chr", "")
    allele_string = f'{row["ref"]}/{row["alt"]}' if row["alt"] != "-" else f'{row["ref"]}/-'
    return f'{chrom} {row["pos"]} {row["end"]} {allele_string} . . .'


def query_vep_batch(variants: list[str]) -> list[dict]:
    delay = VEP_RETRY_BASE_DELAY
    last_exc = None
    for attempt in range(1, VEP_MAX_ATTEMPTS + 1):
        try:
            r = requests.post(
                VEP_REST_URL,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json={"variants": variants},
                timeout=60,
            )
        except requests.RequestException as e:
            last_exc = e
            if attempt < VEP_MAX_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
                continue
            raise

        if r.status_code in VEP_RETRYABLE_STATUS and attempt < VEP_MAX_ATTEMPTS:
            time.sleep(delay)
            delay *= 2
            continue

        r.raise_for_status()
        return r.json()

    # Unreachable in practice (the loop above always returns or raises),
    # but keeps the function's return type honest if VEP_MAX_ATTEMPTS is 0.
    raise last_exc or RuntimeError("VEP REST request failed with no response.")


def extract_annotation(vep_result: dict) -> dict:
    clin_sig = None
    gnomad_af = None
    cosmic_id = None
    rsid = None
    for cv in vep_result.get("colocated_variants", []):
        if cv.get("id", "").startswith("rs"):
            rsid = cv["id"]
        if "clin_sig" in cv:
            clin_sig = ";".join(cv["clin_sig"])
        freqs = cv.get("frequencies")
        if freqs:
            for allele, pops in freqs.items():
                if "gnomade" in pops:
                    gnomad_af = pops["gnomade"]
        cosmic_syns = cv.get("var_synonyms", {}).get("COSMIC")
        if cosmic_syns:
            cosmic_id = ";".join(cosmic_syns)
    return {
        "dbsnp_id": rsid,
        "clnsig": clin_sig,
        "gnomad_af": gnomad_af,
        "cosmic_id": cosmic_id,
    }


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    variants = [build_region_string(row) for _, row in df.iterrows()]
    annotations = []
    for i in range(0, len(variants), BATCH_SIZE):
        batch = variants[i:i + BATCH_SIZE]
        results = query_vep_batch(batch)
        for res in results:
            annotations.append(extract_annotation(res))
        time.sleep(1)  # be polite to the public REST endpoint
    ann_df = pd.DataFrame(annotations)
    out = pd.concat([df.reset_index(drop=True), ann_df], axis=1)
    out["cosmic_hotspot"] = out["cosmic_id"].notna()
    return out


def build_avinput(df: pd.DataFrame, path: str) -> None:
    """Write ANNOVAR's plain-text avinput format: chr start end ref alt row_index."""
    with open(path, "w") as f:
        for idx, row in df.iterrows():
            f.write(f'{row["chrom"]}\t{row["pos"]}\t{row["end"]}\t{row["ref"]}\t{row["alt"]}\t{idx}\n')


def annotate_with_annovar(df: pd.DataFrame) -> pd.DataFrame:
    """Run a real local ANNOVAR install (table_annovar.pl) against refGeneWithVer
    (gene/consequence), ClinVar 2022-12-31, and gnomAD 2.1.1 exome AF.
    Returns a DataFrame of annovar_* columns aligned to df's row order via the
    row-index otherinfo column written by build_avinput.
    """
    table_annovar = os.path.join(ANNOVAR_DIR, "table_annovar.pl")
    if not os.path.exists(table_annovar):
        raise FileNotFoundError(
            f"ANNOVAR not found at {table_annovar}. Expected stage1/annovar/ "
            "with table_annovar.pl and humandb/ populated."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        avinput_path = os.path.join(tmpdir, "patient.avinput")
        out_prefix = os.path.join(tmpdir, "patient")
        build_avinput(df, avinput_path)

        cmd = [
            "perl", table_annovar, avinput_path, "humandb" + os.sep,
            "-buildver", ANNOVAR_BUILD,
            "-out", out_prefix,
            "-remove",
            "-protocol", ANNOVAR_PROTOCOL,
            "-operation", ANNOVAR_OPERATION,
            "-nastring", ".",
            "-otherinfo",
        ]
        result = subprocess.run(cmd, cwd=ANNOVAR_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ANNOVAR failed:\n{result.stderr}")

        multianno_path = f"{out_prefix}.{ANNOVAR_BUILD}_multianno.txt"
        raw = pd.read_csv(multianno_path, sep="\t", dtype=str)

    # The otherinfo columns (from -otherinfo) are appended after the named
    # annotation columns; the last one is the row_index we wrote in build_avinput.
    row_idx = raw.iloc[:, -1].astype(int)

    rename_map = {
        "Func.refGeneWithVer": "annovar_func",
        "Gene.refGeneWithVer": "annovar_gene",
        "GeneDetail.refGeneWithVer": "annovar_gene_detail",
        "ExonicFunc.refGeneWithVer": "annovar_exonic_func",
        "AAChange.refGeneWithVer": "annovar_aa_change",
        "CLNALLELEID": "annovar_clnalleleid",
        "CLNDN": "annovar_clndn",
        "CLNDISDB": "annovar_clndisdb",
        "CLNREVSTAT": "annovar_clnrevstat",
        "CLNSIG": "annovar_clnsig",
        "AF": "annovar_gnomad_af",
        "AF_popmax": "annovar_gnomad_af_popmax",
    }
    keep_cols = [c for c in rename_map if c in raw.columns]
    ann = raw[keep_cols].rename(columns=rename_map)
    ann.index = row_idx.values
    ann = ann.sort_index()
    ann = ann.replace(".", pd.NA)
    return ann.reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/patient/step1_parsed.csv")
    annotated = annotate(df)
    for col in ["clnsig", "gnomad_af", "cosmic_id", "dbsnp_id"]:
        non_null_frac = annotated[col].notna().mean()
        print(f"{col}: {non_null_frac:.1%} non-null")

    print("Running ANNOVAR (local, real) ...")
    annovar_df = annotate_with_annovar(df)
    for col in ["annovar_clnsig", "annovar_gnomad_af", "annovar_exonic_func"]:
        non_null_frac = annovar_df[col].notna().mean()
        print(f"{col}: {non_null_frac:.1%} non-null")
    annotated = pd.concat([annotated.reset_index(drop=True), annovar_df], axis=1)

    print(annotated.head(10).to_string())
    annotated.to_csv("data/patient/step2_annotated.csv", index=False)
