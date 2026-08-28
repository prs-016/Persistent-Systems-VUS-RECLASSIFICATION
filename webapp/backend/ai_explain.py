"""
Plain-language, per-variant explanation of a run's top flagged VUS, via Gemini.

This is a reading aid layered on top of real model output, not a source of
truth: every number it references (scores, bands, tissue expression flags,
gnomAD frequency, MAVE coverage, ClinVar watchlist status) already comes
from the trained pipeline. Gemini is only asked to read those real numbers
for the top few flagged variants and explain, per variant, why each one
was flagged, it is not asked to reclassify anything itself, and it is
explicitly told not to invent signals the pipeline doesn't have (there is
no literature-mining feature in this pipeline, for example). If
GEMINI_API_KEY isn't set, or the request fails, callers get a clear error
and the app keeps working with the table alone, same degrade-gracefully
pattern as the optional Stage 2 timing/direction models.
"""
from __future__ import annotations

import json
import os
import time

import requests

# Gemini's free tier returns 503 (overloaded) or 429 (rate limited) fairly
# often under normal load, both are transient, so a couple of quick retries
# per model clears most of them without the user having to click "Regenerate"
# by hand. If a model is still down after its own retries, fall through to
# the next model in the list rather than give up, gemini-flash-latest being
# overloaded doesn't mean gemini-2.0-flash is too, they're separate capacity
# pools. A non-retryable error (bad request, model retired, etc.) skips
# straight to the next candidate without wasting the retry budget on it.
RETRYABLE_STATUS = {429, 503}
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.5

# How many of the flagged variants get a deep individual explanation. Kept
# small deliberately, this is meant to be read, not skimmed, one paragraph
# per variant for the handful most worth a human's attention first.
TOP_N = 3

# First entry wins normally; GEMINI_MODEL lets you pin one model instead of
# falling back (set it to a single model name to disable fallback entirely).
# Verified live against the real key: gemini-2.0-flash and gemini-2.5-flash
# are both retired (404, Google's own error names gemini-3.6-flash as their
# replacement), gemini-pro-latest hit a quota limit on the free tier, so
# those three are deliberately left out rather than wasting a request on a
# candidate that's already known to fail.
_env_model = os.environ.get("GEMINI_MODEL", "").strip()
CANDIDATE_MODELS = [_env_model] if _env_model else [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.6-flash",
]


def _url_for(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class AiExplainError(Exception):
    pass


def _fmt_pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "unknown"


def _variant_key(v: dict) -> str:
    return f"{v.get('chrom')}:{v.get('pos')}{v.get('ref')}>{v.get('alt')}"


def _score_sort_key(v: dict):
    score = v.get("stage2_score")
    # ClinVar-watchlist-sourced rows and generalizable rows both use
    # stage2_score for the deep-dive ranking, higher first; unscored rows
    # (score missing) sort last rather than crashing the sort.
    return score if isinstance(score, (int, float)) else -1.0


def select_top_variants(flagged: list[dict], limit: int = TOP_N) -> list[dict]:
    """Rank the flagged variants by reclassification score and take the
    top `limit`. This is the set that gets a deep, individual explanation,
    rather than every flagged variant getting one shallow sentence."""
    return sorted(flagged, key=_score_sort_key, reverse=True)[:limit]


def _variant_fact_block(v: dict, idx: int) -> str:
    gene = v.get("gene") or "unknown gene"
    change = f"{v.get('chrom')}:{v.get('pos')} {v.get('ref')}>{v.get('alt')}"
    full_name = v.get("variant_label") or change
    variant_type = v.get("variant_type")
    hgvs_c = v.get("hgvs_c")
    hgvs_p = v.get("hgvs_p")
    transcript = v.get("transcript")
    if hgvs_c:
        coding_note = f"{hgvs_c}" + (f" ({hgvs_p})" if hgvs_p else "") + (f" on {transcript}" if transcript else "")
    else:
        coding_note = "no coding-transcript change reported (likely non-exonic)"
    type_note = variant_type or "consequence not called by the annotation pipeline"
    source = v.get("stage2_source")
    source_note = (
        "matched against the project's own ClinVar VUS watchlist (a real prior "
        "resolution-probability estimate for this exact variant)"
        if source == "clinvar_v12"
        else "scored by the generalizable model (no prior ClinVar history for "
        "this exact variant, scored from its own annotation-derived features)"
    )
    band = v.get("stage2_band") or "unbanded"
    score = v.get("stage2_score")
    score_note = f"{score:.3f}" if isinstance(score, (int, float)) else "unknown"
    origin = v.get("predicted_class") or "unresolved"
    origin_conf = v.get("predicted_class_confidence")
    origin_note = f" ({_fmt_pct(origin_conf)} confidence)" if origin_conf is not None else ""
    tissue_level = v.get("hpa_expression_level") or "unknown"
    low_tissue = v.get("low_tissue_expression_flag")
    tissue_note = "low breadth" if low_tissue else "typical breadth"
    gnomad_af = v.get("gnomad_af")
    gnomad_note = f"{gnomad_af:.2e}" if isinstance(gnomad_af, (int, float)) else "not observed in gnomAD"
    mave = v.get("has_mave_coverage")
    mave_note = "has functional assay (MAVE) coverage" if mave else "no MAVE assay coverage in this pipeline's data"
    cosmic = v.get("cosmic_hotspot")
    cosmic_note = "flagged as a COSMIC hotspot position" if cosmic else "not a COSMIC hotspot position"
    direction = v.get("direction_pathogenic_probability")
    direction_note = f"{_fmt_pct(direction)} probability of resolving pathogenic if/when it resolves" if direction is not None else "no direction estimate available"
    p10y = v.get("p_resolved_by_10y")
    p10y_note = f"{_fmt_pct(p10y)} probability of resolving within 10 years" if p10y is not None else "no 10-year resolution estimate available"

    return "\n".join([
        f"Variant {idx} [{change}]:",
        f"- Gene: {gene}",
        f"- Full variant name: {full_name}",
        f"- Coding/protein change: {coding_note}",
        f"- Variant type (consequence): {type_note}",
        f"- Reclassification score: {score_note}, band '{band}', {source_note}",
        f"- Stage 1 predicted origin: {origin}{origin_note}",
        f"- Tissue expression breadth (HPA, gene-level, not sample-specific): {tissue_level} ({tissue_note})",
        f"- gnomAD population allele frequency: {gnomad_note}",
        f"- Functional assay coverage: {mave_note}",
        f"- Somatic hotspot signal: {cosmic_note}",
        f"- Resolution direction: {direction_note}",
        f"- Resolution timing: {p10y_note}",
    ])


def build_prompt(summary: dict, top_variants: list[dict], tissue: str) -> str:
    variant_blocks = "\n\n".join(
        _variant_fact_block(v, i + 1) for i, v in enumerate(top_variants)
    )
    variant_keys = [_variant_key(v) for v in top_variants]

    return "\n".join([
        "You are a genomics triage assistant reading the output of an already-run "
        "classification pipeline for a cancer genomics project, writing a short "
        "briefing per variant for someone about to manually review it. Do not "
        "use em dashes.",
        "",
        "Every number under \"Facts for each variant\" below (score, band, "
        "tissue expression, gnomAD frequency, assay coverage, origin call, "
        "resolution timing/direction) is real model output from this pipeline, "
        "never invent or alter one of those numbers. This pipeline has no "
        "literature-mining, citation, or real submission-count feature, do not "
        "claim it does or cite specific studies, case counts, or submitter "
        "activity as if the pipeline tracked them.",
        "",
        f"Sample tissue context selected for this run: {tissue}.",
        f"Run totals: {summary.get('total_variants')} variants evaluated, "
        f"{summary.get('flagged_for_reclassification_review')} flagged for "
        "reclassification review in total; the variants below are the "
        f"{len(top_variants)} highest-scoring of those flagged variants.",
        "",
        "Facts for each variant:",
        "",
        variant_blocks,
        "",
        "For each variant, write exactly three short paragraphs (2-3 sentences "
        "each), separated by one blank line, in this order:",
        "",
        "1. Gene and variant. Name the gene and, briefly, what it's generally "
        "known for biologically (e.g. tumor suppressor, DNA mismatch repair, "
        "cell-cycle regulator), using your own general knowledge, not pipeline "
        "output. Then state the variant's full name (coding and protein "
        "change, or say plainly it's non-exonic with no protein change) and "
        "its type (missense, frameshift, nonsense, intronic, and so on).",
        "2. Evidence. Explain, using only the real facts listed above, which "
        "specific signals drove this variant's reclassification flag, "
        "referencing the actual score, band, tissue expression breadth, "
        "gnomAD frequency, assay coverage, origin call, and ClinVar watchlist "
        "status where each is relevant. Do not restate every fact, pick the "
        "ones that actually drive the flag.",
        "3. Outlook. State the model's resolution-timing and direction "
        "estimates in plain language (e.g. roughly a NN% chance of resolving "
        "within 10 years, and if it does, a NN% chance that resolution is "
        "pathogenic), using the real numbers above, or say plainly that no "
        "timing or direction estimate is available for this variant if both "
        "are missing. Make clear this is a statistical estimate from patterns "
        "in previously resolved variants, not a literature review or a count "
        "of new submissions, since this pipeline doesn't have either. Do not "
        "give a diagnosis or state a final reclassification.",
        "",
        "Respond with ONLY a JSON object, no markdown fencing, no commentary "
        "outside the JSON, in exactly this shape:",
        '{"explanations": [{"variant": "<the bracketed variant id exactly as given, '
        'e.g. ' + (variant_keys[0] if variant_keys else "chr1:12345A>G") + '>", '
        '"explanation": "<the three paragraphs for that variant, separated by '
        'a blank line>"}]}',
        f"The array must have exactly {len(top_variants)} entries, one per variant "
        "listed above, in the same order.",
    ])


def _call_model(model: str, api_key: str, prompt: str) -> str | None:
    """Try one model with its own retry budget. Returns the response text on
    success, or None if this model should be skipped in favor of the next
    candidate (exhausted retries on a transient error, or a non-retryable
    error like a retired/unknown model). Raises AiExplainError only for
    problems that would affect every model equally (network unreachable)."""
    url = _url_for(model)
    resp = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(
                url,
                params={"key": api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.25,
                        # Some of these models (gemini-flash-latest, gemini-3.6-flash)
                        # spend part of maxOutputTokens on internal "thinking"
                        # tokens before the real answer, verified up to ~500 in
                        # testing; each variant now gets three paragraphs
                        # instead of one, so 2800 leaves comfortable headroom
                        # for that plus JSON structure overhead.
                        # thinkingConfig isn't used here, it's accepted by some
                        # of these model names and rejected with a 400 by
                        # others, not worth the per-model special-casing.
                        "maxOutputTokens": 2800,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=45,
            )
        except requests.RequestException:
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            return None  # let the next candidate model try

        if resp.status_code in RETRYABLE_STATUS:
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            return None  # exhausted retries on this model, try the next one
        break

    if resp is None or resp.status_code != 200:
        return None  # non-retryable error (bad request, retired model, ...)

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return None

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    return text or None


def _parse_explanations(raw_text: str, top_variants: list[dict]) -> list[dict] | None:
    """Parse the model's JSON reply and zip each explanation back onto its
    real variant record (so the frontend gets real gene/position/score/band
    data alongside the generated prose, not just whatever the model echoed
    back). Returns None if the reply doesn't parse or doesn't line up,
    letting the caller fall through to the next candidate model."""
    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None

    items = parsed.get("explanations") if isinstance(parsed, dict) else None
    if not isinstance(items, list) or not items:
        return None

    by_key = {_variant_key(v): v for v in top_variants}
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        explanation = (item.get("explanation") or "").strip()
        if not explanation:
            continue
        variant = by_key.get(item.get("variant"))
        if variant is None:
            # Fall back to positional matching if the model altered the key
            # formatting slightly but kept the order intact.
            idx = len(out)
            if idx < len(top_variants):
                variant = top_variants[idx]
        if variant is None:
            continue
        out.append({
            "gene": variant.get("gene"),
            "chrom": variant.get("chrom"),
            "pos": variant.get("pos"),
            "ref": variant.get("ref"),
            "alt": variant.get("alt"),
            "variant_type": variant.get("variant_type"),
            "hgvs_c": variant.get("hgvs_c"),
            "hgvs_p": variant.get("hgvs_p"),
            "variant_label": variant.get("variant_label")
            or f"{variant.get('chrom')}:{variant.get('pos')} {variant.get('ref')}>{variant.get('alt')}",
            "stage2_band": variant.get("stage2_band"),
            "stage2_score": variant.get("stage2_score"),
            "explanation": explanation,
        })

    return out or None


def generate_explanation(summary: dict, flagged: list[dict], tissue: str) -> list[dict]:
    """Returns a list of per-variant explanation dicts for the top flagged
    VUS by reclassification score (see TOP_N). Empty list if nothing in
    this run was flagged."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise AiExplainError(
            "AI explanation is not configured. Set GEMINI_API_KEY in webapp/backend/.env "
            "and restart the server."
        )

    top_variants = select_top_variants(flagged)
    if not top_variants:
        return []

    prompt = build_prompt(summary, top_variants, tissue)
    for model in CANDIDATE_MODELS:
        raw_text = _call_model(model, api_key, prompt)
        if not raw_text:
            continue
        explanations = _parse_explanations(raw_text, top_variants)
        if explanations:
            return explanations

    raise AiExplainError(
        "Gemini is unavailable right now (tried "
        f"{', '.join(CANDIDATE_MODELS)}, all overloaded, erroring, or returned "
        "output that couldn't be parsed). This is usually temporary, try "
        "Regenerate again shortly."
    )
