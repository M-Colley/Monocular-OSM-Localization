"""scripts/eval_safevpr.py: the registered hypotheses are computed literally.

Uses a stand-in for SafeVPR's conformal module (accept = score >= one tau) and
metrics module, so no model and no SafeVPR checkout are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_safevpr as E  # noqa: E402

_conformal = SimpleNamespace(
    fit_mondrian_ltt=lambda s, y, alpha: {"edges": np.array([-np.inf, np.inf]),
                                          "taus": np.array([float(np.quantile(s, 0.8))])},
    accept=lambda s, thr: np.asarray(s) >= thr["taus"][0])


def _auroc(scores, labels):
    from scipy.stats import mannwhitneyu
    pos, neg = scores[labels == 1], scores[labels == 0]
    return mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


_metrics = SimpleNamespace(auroc=_auroc)


def _pairs(fleet, n, good_patch=True, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ok = i % 5 == 0
        out.append({"clip": f"{fleet}{i % 3}", "fleet": fleet, "correct": ok,
                    "patch_mnn": (0.8 if ok else 0.2) + rng.normal(0, 0.01) if good_patch
                    else rng.random(),
                    "cosine": rng.random()})
    return out


def test_hypotheses_follow_the_registered_wording() -> None:
    res = E.evaluate(_pairs("core", 100) + _pairs("overlay", 100, seed=1), _conformal, _metrics)
    pm, co = res["scores"]["patch_mnn"]["test"], res["scores"]["cosine"]["test"]
    assert pm["fdr"] == 0.0 and pm["tpr"] == 1.0          # a separating score: valid, full recall
    assert res["hypotheses"]["S1"] is True
    assert res["hypotheses"]["S2"] is (co["fdr"] is not None and co["fdr"] > 0.0)
    assert res["hypotheses"]["S3"] is True
    assert set(res["scores"]["patch_mnn"]["test_per_clip"]) == {"overlay0", "overlay1", "overlay2"}


def test_a_useless_score_fails_s1() -> None:
    res = E.evaluate(_pairs("core", 200, good_patch=False) + _pairs("overlay", 200, False, 3),
                     _conformal, _metrics)
    assert res["hypotheses"]["S1"] is False
