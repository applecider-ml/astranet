"""SkyPortal analysis-service adapter: the science half of an ASTRANet analysis job.

`analyze()` takes one or more spectra already parsed from a SkyPortal payload (wavelengths
and fluxes in the observed frame), the source redshift if SkyPortal has one, and the service
parameters, and returns the dictionary a SkyPortal analysis callback carries: results, flat
annotations and classification rows. The bridge that talks to SkyPortal only parses the
payload and calls this, so the logic is versioned here with the models it uses.

Parameters (all optional): redshift (overrides the source value), use_redshift (True; when
False the spectrum stays in the observed frame and the redshift token is withheld),
models_dir (default $ASTRANET_MODELS or ./models), novelty (path of the novelty bundle; default
$ASTRANET_NOVELTY or <models_dir>/novelty_v1.joblib; "off" disables the stage).

The novelty stage (astranet.novelty) scores the object against ordinary objects of the same redshift
stratum and reports a p-value and alarms at 1, 5, 10 and 15 %. Novae are ordinary for this stage: they
are in its reference set and do not count as novel. The stage is skipped, with a note in the results,
when the bundle or its dependencies (scikit-learn, joblib, luna-ood) are missing.

Command line, for one exported spectrum:

    python -m astranet.skyportal spectrum.dat --redshift 0.05
"""
from __future__ import annotations

import json
import math
import os

SKYPORTAL_TAXONOMY = "Sitewide Taxonomy"
SKYPORTAL_ORIGIN = "ASTRANet"
# ASTRANet's seven classes -> the nearest label in SkyPortal's Sitewide Taxonomy. SLSN has no
# umbrella node there (only subtypes), so it is reported in the annotations only.
ASTRANET_TO_TAXONOMY = {"Ia": "Ia", "Ibc": "Ib/c", "II": "Type II", "AGN": "AGN", "CV": "Cataclysmic",
                        "TDE": "Tidal Disruption Event", "SLSN": None}
_ENSEMBLE = {}
_NOVELTY = {}


def _to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _ensemble(models_dir):
    if models_dir not in _ENSEMBLE:
        from .predict import Ensemble
        _ENSEMBLE[models_dir] = Ensemble.from_dir(models_dir)
    return _ENSEMBLE[models_dir]


def _novelty(path):
    if path not in _NOVELTY:
        from .novelty import Novelty
        _NOVELTY[path] = Novelty.load(path)
    return _NOVELTY[path]


def analyze(spectra, redshift=None, params: dict | None = None, resource_id: str = "obj") -> dict:
    """`spectra`: list of dicts with `wavelengths`, `fluxes` and optionally `id`, `observed_at`, `instrument`."""
    import numpy as np
    from .preprocess import SpectrumRejected
    params = dict(params or {})
    z = _to_float(params.get("redshift")); z = _to_float(redshift) if z is None else z
    if str(params.get("use_redshift", True)).lower() in ("false", "0", "no"):
        z = None
    models_dir = params.get("models_dir") or os.environ.get("ASTRANET_MODELS", "models")
    ens = _ensemble(models_dir)
    nov_path = params.get("novelty") or os.environ.get("ASTRANET_NOVELTY") or os.path.join(models_dir, "novelty_v1.joblib")
    feats, grids = [], []
    rows = []
    for s in spectra:
        row = {k: s.get(k) for k in ("id", "observed_at", "instrument")}
        try:
            out = ens.predict(s["wavelengths"], s["fluxes"], redshift=z, return_features=True)
            feats.append(out.pop("_features")); grids.append(out.pop("_grid")); row.update(status="ok", **out)
        except SpectrumRejected as e:
            row.update(status="rejected", reason=str(e))
        rows.append(row)
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return dict(status="failure", message="no spectrum could be placed on the model grid", results=dict(spectra=rows))
    P = np.mean([[r["probabilities"][c] for c in ens.classes] for r in ok], axis=0)
    top = int(P.argmax()); cls = ens.classes[top]
    obj = dict(predicted=cls, probability=float(P[top]), probabilities={c: float(v) for c, v in zip(ens.classes, P)},
               n_spectra=len(ok), n_rejected=len(rows) - len(ok), redshift_used=z, frame="rest" if z is not None and -1 < z <= 0.87 else "observed",
               spectra_agreeing=int(sum(r["predicted"] == cls for r in ok)))
    annotations = {"astranet_class": cls, "astranet_prob": round(float(P[top]), 4), "astranet_n_spectra": len(ok),
                   "astranet_frame": obj["frame"], **{f"astranet_p_{c}": round(float(v), 4) for c, v in zip(ens.classes, P)}}
    if str(nov_path).lower() in ("off", "none", "false"):
        novelty = dict(status="off")
    elif not os.path.exists(nov_path):
        novelty = dict(status="unavailable", reason=f"no novelty bundle at {nov_path}")
    else:
        try:
            novelty = dict(status="ok", **_novelty(nov_path).score(feats, np.stack(grids), z))
        except ImportError as e:
            novelty = dict(status="unavailable", reason=f"missing dependency: {e.name}")
    message = f"ASTRANet: {cls} ({P[top]:.2f}) from {len(ok)} spectra"
    if novelty["status"] == "ok":
        annotations.update(astranet_novelty_p=round(novelty["p_value"], 4), astranet_novelty_alarm=novelty["alarm_rate"] or "none",
                           astranet_novelty_stratum=novelty["stratum"])
        if novelty["alarm_rate"]:
            message += f"; novelty alarm at {novelty['alarm_rate']} (p = {novelty['p_value']:.3f})"
    label = ASTRANET_TO_TAXONOMY.get(cls)
    classifications = [] if label is None else [dict(obj_id=resource_id, classification=label, probability=round(float(P[top]), 4),
                                                     taxonomy=SKYPORTAL_TAXONOMY, origin=SKYPORTAL_ORIGIN, ml=True)]
    return dict(status="success", message=message,
                results=dict(object=obj, novelty=novelty, spectra=rows), annotations=annotations, classifications=classifications)


def main(argv=None):
    import argparse
    from .preprocess import read_dat
    ap = argparse.ArgumentParser(description="ASTRANet on exported SkyPortal spectra")
    ap.add_argument("files", nargs="+"); ap.add_argument("--redshift", type=float); ap.add_argument("--models-dir"); ap.add_argument("--novelty", help='novelty bundle, or "off"')
    ap.add_argument("--observed-frame", action="store_true", help="withhold the redshift")
    a = ap.parse_args(argv)
    spectra = []
    for f in a.files:
        w, fl = read_dat(f); spectra.append(dict(id=os.path.basename(f), wavelengths=w, fluxes=fl))
    out = analyze(spectra, a.redshift, dict(models_dir=a.models_dir, novelty=a.novelty, use_redshift=not a.observed_frame))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
