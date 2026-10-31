"""Análise estatística pareada (metodologia, Seção "Análise estatística").

Unidade estatística = réplica (nunca o item). Para as políticas a (referência)
e b (tratamento), na mesma réplica e com números aleatórios comuns:

    D_r = Y_{r,b} − Y_{r,a}

  * Escolha do teste pela distribuição de D: t pareado quando D é
    aproximadamente simétrica e sem valores extremos; caso contrário,
    Wilcoxon de postos sinalizados com IC por bootstrap pareado.
  * Magnitude: d_z = D̄/s_D (t) ou correlação bisserial de postos
    (Wilcoxon), e Δ% = 100·(Ȳ_b − Ȳ_a)/Ȳ_a.
  * Relevância prática por margem δ: melhoria relevante / equivalência
    prática / inconclusivo, pelo IC da diferença.
  * Não inferioridade com margem explícita (não se interpreta teste não
    significativo como igualdade).
  * Comparações múltiplas: Holm dentro de cada família.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

ALPHA = 0.05


def _clean(a, b) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]


def _bootstrap_ci(d: np.ndarray, alpha: float, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_compare(
    ref, treat, better: str = "lower", alpha: float = ALPHA, delta_rel: float | None = None,
) -> dict:
    """Compara tratamento vs referência; `better`='lower' (LT, WT, WIP) ou 'higher' (TH, FE).

    p_value é unilateral na direção da melhoria.
    """
    a, b = _clean(ref, treat)
    n = len(a)
    if n < 3:
        return {"n": n, "test": None, "p_value": 1.0, "note": "réplicas insuficientes"}
    d = b - a
    mean_a, mean_b, dbar = float(a.mean()), float(b.mean()), float(d.mean())
    sd = float(d.std(ddof=1))
    alt = "less" if better == "lower" else "greater"
    out = {
        "n": n, "mean_ref": mean_a, "mean_treat": mean_b, "mean_diff": dbar, "sd_diff": sd,
        "delta_pct": 100.0 * (mean_b - mean_a) / mean_a if mean_a != 0 else None,
        "better": better,
    }
    scale = max(abs(mean_a), abs(mean_b), 1e-12)
    if sd <= 1e-9 * scale or np.allclose(d, 0, atol=1e-9 * scale):
        out.update(test="degenerate", p_value=1.0 if dbar == 0 or (dbar > 0) == (better == "lower") else 0.0,
                   ci=[dbar, dbar], effect=None, effect_name=None,
                   note="diferenças constantes (configurações equivalentes)")
    else:
        skew = float(stats.skew(d))
        q1, q3 = np.quantile(d, [0.25, 0.75])
        iqr = q3 - q1
        extreme = bool(np.any((d < q1 - 3 * iqr) | (d > q3 + 3 * iqr))) if iqr > 0 else False
        if abs(skew) < 1.0 and not extreme:
            t = stats.ttest_1samp(d, 0.0, alternative=alt)
            h = stats.t.ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n)
            out.update(test="t pareado", p_value=float(t.pvalue), ci=[dbar - h, dbar + h],
                       effect=dbar / sd, effect_name="d_z", skewness=skew)
        else:
            w = stats.wilcoxon(d, alternative=alt, zero_method="wilcox")
            nz = d[d != 0]
            ranks = stats.rankdata(np.abs(nz))
            rbc = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
            out.update(test="Wilcoxon (postos sinalizados)", p_value=float(w.pvalue),
                       ci=list(_bootstrap_ci(d, alpha)), effect=rbc,
                       effect_name="correlação bisserial de postos", skewness=skew)
    if delta_rel is not None:
        out["practical"] = practical_relevance(out, delta_rel * abs(mean_a), better)
        out["delta_rel"] = delta_rel
    return out


def practical_relevance(res: dict, delta_abs: float, better: str) -> str:
    lo, hi = res["ci"]
    if better == "higher":
        lo, hi = -hi, -lo          # converte para "menor é melhor"
    if hi < -delta_abs:
        return "melhoria relevante"
    if lo > delta_abs:
        return "piora relevante"
    if lo >= -delta_abs and hi <= delta_abs:
        return "equivalência prática"
    return "inconclusivo"


def non_inferiority(ref, treat, margin_rel: float, better: str = "higher", alpha: float = ALPHA) -> dict:
    """H0: tratamento pior que a referência por mais que a margem.

    Para métrica 'higher' (vazão): H0: D̄ ≤ −m·Ȳ_ref  vs  H1: D̄ > −m·Ȳ_ref.
    """
    a, b = _clean(ref, treat)
    if len(a) < 3:
        return {"p_value": 1.0, "non_inferior": False, "note": "réplicas insuficientes"}
    m = margin_rel * abs(float(a.mean()))
    d = b - a
    shifted = d + m if better == "higher" else -(d - m)
    sd = float(shifted.std(ddof=1))
    if sd <= 1e-9 * max(abs(float(a.mean())), 1e-12):
        p = 0.0 if shifted.mean() > 0 else 1.0
    else:
        p = float(stats.ttest_1samp(shifted, 0.0, alternative="greater").pvalue)
    return {"margin_rel": margin_rel, "margin_abs": m, "mean_diff": float(d.mean()),
            "p_value": p, "non_inferior": p < alpha}


def half_width(d, alpha: float = ALPHA) -> float:
    """Meia largura do IC da diferença média pareada (metodologia, Eq.
    `eq:meia_largura_ic`): h = t_{1−α/2, R−1} · s_D / √R."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("inf")
    return float(stats.t.ppf(1 - alpha / 2, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))


def required_replications(d, h_target: float, alpha: float = ALPHA, r_max: int = 1000) -> int:
    """Menor R com t_{1−α/2,R−1}·s_D/√R ≤ h_target, usando s_D do piloto.
    A regra é fixada antes de olhar o resultado (não se para cedo por
    resultado favorável)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return r_max
    sd = float(d.std(ddof=1))
    if sd == 0 or h_target <= 0:
        return len(d) if sd == 0 else r_max
    for r in range(2, r_max + 1):
        if stats.t.ppf(1 - alpha / 2, r - 1) * sd / math.sqrt(r) <= h_target:
            return r
    return r_max


def holm(pvals: list[float]) -> list[float]:
    """p-valores ajustados por Holm (sequencial, controla FWER)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def describe(values) -> dict:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return {}
    n = len(v)
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    return {"mean": float(v.mean()), "median": float(np.median(v)), "sd": float(v.std(ddof=1)) if n > 1 else 0.0,
            "ci95": [float(v.mean() - h), float(v.mean() + h)], "n": n}
