"""Config block for all Stage 1 thresholds — edit here, not in logic."""

GERMLINE_VAF_MIN = 0.35
GERMLINE_VAF_MAX = 0.65
GERMLINE_VAF_HOM_MIN = 0.90

GNOMAD_COMMON_AF_THRESHOLD = 0.01

# Real categories found in the current proteinatlas.tsv bulk file's
# "RNA tissue distribution" column: Detected in all/many/some/single, Not detected.
# The old per-tissue normal_tissue.tsv (Gene x Tissue x Level x Reliability) is no
# longer distributed by HPA (confirmed 404 on all known URLs), so this flag is a
# gene-level "expressed anywhere assayed" signal, not a lookup for one specific
# tissue. Stated explicitly wherever this flag is used.
HPA_LOW_EXPRESSION_CATEGORIES = {"Not detected"}
HPA_RELIABLE_CATEGORIES = {"Approved", "Supported", "Enhanced"}

# ClinVar CLNSIG terms (VEP REST lowercases and underscores these, e.g.
# "likely_pathogenic"). A variant only gets a confident Pathogenic/Benign
# call if every term in its (possibly multi-term, ";"-joined) CLNSIG value
# is in the matching set below — a mixed/conflicting CLNSIG (e.g.
# "uncertain_significance;likely_benign;pathogenic") is NOT confident and
# falls through to the classifier as VUS, same as a missing CLNSIG.
CLINVAR_PATHOGENIC_TERMS = {"pathogenic", "likely_pathogenic"}
CLINVAR_BENIGN_TERMS = {"benign", "likely_benign"}
