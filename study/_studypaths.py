"""Roots of the study. Override with environment variables when the study runs outside the repository."""
import os
from pathlib import Path
STUDY = Path(os.environ.get('ASTRANET_STUDY', Path(__file__).resolve().parent))
REPO = Path(os.environ.get('ASTRANET_REPO', STUDY.parent))
# Copies of this study run on the single-backbone caches (study_tcn: tcn6, study_sn: spectranet6), for the
# backbone comparison; VARIANTS_ROOT/<name> is where decide.py and lifetime_plife.py look for them.
VARIANTS_ROOT = Path(os.environ.get('ASTRANET_STUDY_VARIANTS', STUDY.parent))
