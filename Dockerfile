# ASTRANet inference image: the package, the twelve checkpoints and the novelty bundle, so jobs run offline.

FROM python:3.12-slim-bookworm

ARG WEIGHTS_URL=https://github.com/applecider-ml/astranet/releases/download/v1.0.0/astranet_checkpoints.tar.gz
ARG NOVELTY_URL=https://github.com/applecider-ml/astranet/releases/download/v1.0.0/novelty.joblib

ENV PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    ASTRANET_MODELS=/opt/astranet/models

WORKDIR /src
COPY . .

# CPU build of torch; scikit-learn is pinned to the version that wrote the novelty bundle; requests is for the
# analysis-callback upload. The weights are downloaded from the release and checked against their MD5.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[novelty]" "scikit-learn==1.8.0" requests \
    && mkdir -p /opt/astranet \
    && python tools/fetch_weights.py --url "$WEIGHTS_URL" --novelty-url "$NOVELTY_URL" --dest /opt/astranet \
    && rm -f /opt/astranet/astranet_checkpoints.tar.gz \
    && rm -rf /src

# Fail the build if the ensemble or the novelty stage does not load.
RUN python -c "from astranet.predict import Ensemble; from astranet.novelty import Novelty; import astranet.report, matplotlib; Ensemble.from_dir('/opt/astranet/models', device='cpu'); Novelty.load('/opt/astranet/models/novelty.joblib'); print('astranet ok')"

WORKDIR /work
CMD ["python", "-m", "astranet.skyportal", "--help"]
