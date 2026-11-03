"""
verify.py -- fail loudly. Must not modify any artifact.

Exits non-zero if any check fails, printing one actionable line per failure.
This script is the only thing that turns the pipeline contract in 01_system.md
from prose into an enforced constraint. A review report may not claim numbers are
consistent without pasting this script's output (Rule 8).

Usage:  python verify.py [--run RUN_ID]
Exit:   0 = all checks pass, 1 = at least one failure
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

BANNED_IN_VISUALIZE = [
    "sklearn", "torch", "tensorflow", "joblib", "scipy.ndimage",
    "median_filter", "gaussian_filter", "train_test_split",
    ".fit(", ".predict(", "decision_function", "import models", "import dataset",
]

# Literals a manuscript is allowed to print without being a metric (Rule 7).
STRUCTURAL_PATTERNS = [
    r"^.*?\\begin\{document\}",
    r"\\label\{[^}]*\}", r"\\ref\{[^}]*\}", r"\\cite\{[^}]*\}", r"\\eqref\{[^}]*\}",
    r"\\newcommand\{[^}]*\}\{[^}]*\}", r"\\includegraphics\[[^\]]*\]",
    r"\\begin\{[^}]*\}", r"\\end\{[^}]*\}", r"\\input\{[^}]*\}",
    r"\\usepackage(\[[^\]]*\])?\{[^}]*\}", r"\\documentclass(\[[^\]]*\])?\{[^}]*\}",
    r"\$[^$]*\$", r"\\\[.*?\\\]", r"(?<!\\)%[^\n]*",
    r"\\(vspace|hspace|setlength|columnsep|textwidth|linewidth|baselineskip)[^\n]*",
]


class Checker:
    def __init__(self):
        self.failures = []
        self.passes = []

    def ok(self, name, detail=""):
        self.passes.append(f"PASS  {name}" + (f"  --  {detail}" if detail else ""))

    def fail(self, name, detail):
        self.failures.append(f"FAIL  {name}\n        {detail}")

    def report(self):
        print("=" * 78)
        for p in self.passes:
            print(p)
        if self.failures:
            print("-" * 78)
            for f in self.failures:
                print(f)
        print("=" * 78)
        print(f"{len(self.passes)} passed, {len(self.failures)} failed")
        return 1 if self.failures else 0


def strip_structural(tex):
    for pat in STRUCTURAL_PATTERNS:
        tex = re.sub(pat, " ", tex, flags=re.S)
    return tex


def collect_json_numbers(run):
    """Every numeric value stored anywhere under runs/<run>/metrics, flattened."""
    vals = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            vals.add(float(o))

    for p in glob.glob(os.path.join(run, "metrics", "*.json")):
        with open(p) as f:
            walk(json.load(f))
    return vals


def close_to_any(x, pool, tol=0.011):
    return any(abs(x - v) <= tol for v in pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    ck = Checker()

    latest = os.path.join(ROOT, "runs", "LATEST")
    if not os.path.exists(latest):
        print("FAIL  no runs/LATEST -- nothing to verify. Run train.py first.")
        return 1
    run_id = args.run or open(latest).read().strip()
    run = os.path.join(ROOT, "runs", run_id)
    print(f"verifying run: {run_id}\n")

    # ---- 1. manifest + label validation ---------------------------------
    mf = os.path.join(run, "manifest.json")
    if not os.path.exists(mf):
        ck.fail("manifest exists", f"{mf} missing")
    else:
        man = json.load(open(mf))
        need = ["seed", "config_sha", "code_sha", "intent"]
        miss = [k for k in need if k not in man]
        if miss:
            ck.fail("manifest complete", f"missing keys: {miss}")
        else:
            ck.ok("manifest complete", f"seed={man['seed']}, config_sha={man['config_sha']}")

    lvp = os.path.join(run, "logs", "label_validation.json")
    if not os.path.exists(lvp):
        ck.fail("label validation ran", f"{lvp} missing")
    else:
        lv = json.load(open(lvp))
        n = lv.get("n_defects_with_positive_contrast")
        ck.ok("label validation ran",
              f"{n}/9 defects show positive local contrast at the calibrated geometry")

    # ---- 2. visualize.py computes nothing (Rule 3) -----------------------
    vp = os.path.join(ROOT, "visualize.py")
    src = open(vp).read()
    hits = [b for b in BANNED_IN_VISUALIZE if b in src]
    if hits:
        ck.fail("visualize.py imports no model/compute library",
                f"banned symbols present: {hits}")
    else:
        ck.ok("visualize.py imports no model/compute library",
              f"{len(BANNED_IN_VISUALIZE)} banned symbols absent")

    # ---- 3. no hardcoded metric literals in source (Rule 1) --------------
    offenders = []
    for py in sorted(glob.glob(os.path.join(ROOT, "*.py"))):
        if os.path.basename(py) in ("verify.py", "diagnose_geometry.py",
                                   "calibrate_geometry.py"):
            continue  # one-off diagnostics, not pipeline stages that report metrics
        for i, line in enumerate(open(py), 1):
            code = line.split("#")[0]
            if "np.linspace" in code or "range(" in code:
                continue
            floats = [float(x) for x in re.findall(r"(?<![\w.])(\d{2,3}\.\d{2})(?![\d])", code)]
            metricish = [f for f in floats if 30.0 <= f <= 100.0]
            if metricish:
                offenders.append(f"{os.path.basename(py)}:{i}: {metricish} in `{code.strip()[:70]}`")
    if offenders:
        ck.fail("no hardcoded metric literals in source", "\n        ".join(offenders))
    else:
        ck.ok("no hardcoded metric literals in source")

    # ---- 4. splits are actually leakage-free (the core claim) ------------
    from scipy import ndimage  # verify.py is allowed to compute; visualize.py is not
    cfg = json.load(open(os.path.join(ROOT, "configs", "config.json")))
    pp = cfg["postprocess"]
    reach = int(pp["patch_radius"]) if "patch_radius" in pp else (
        (cfg["models"]["patch_size"] // 2)
        + (pp["median_size"] // 2)
        + int(pp.get("gaussian_truncate", 4.0) * pp["gaussian_sigma"] + 0.5))
    bad = []
    checked = 0
    for mp in sorted(glob.glob(os.path.join(run, "masks", "*.npz"))):
        tag = os.path.basename(mp)[:-4]
        z = np.load(mp)
        fit, test = z["fit_mask"], z["test_mask"]
        grown = ndimage.binary_dilation(
            test, ndimage.generate_binary_structure(2, 2), iterations=reach)
        overlap = int((grown & fit).sum())
        if tag.startswith("random_optimistic"):
            if overlap == 0:
                bad.append(f"{tag}: expected adjacency leakage (it is the optimistic "
                           f"bound) but found none")
            checked += 1
            continue
        if overlap > 0:
            bad.append(f"{tag}: {overlap} training pixels lie within {reach} px of a test "
                       f"pixel -- guard band is too small")
        checked += 1
    if bad:
        ck.fail("primary/secondary splits are leakage-free", "\n        ".join(bad))
    else:
        ck.ok("primary/secondary splits are leakage-free",
              f"{checked} splits checked at {reach} px reach; random split confirmed "
              f"leaky by construction")

    # ---- 5. metrics recompute from stored predictions --------------------
    pf_path = os.path.join(run, "metrics", "per_fold.json")
    if not os.path.exists(pf_path):
        ck.fail("per_fold.json exists", "run evaluate.py first")
        return ck.report()
    per_fold = json.load(open(pf_path))

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    mism, n_re = [], 0
    for pnpz in sorted(glob.glob(os.path.join(run, "predictions", "*.npz"))):
        base = os.path.basename(pnpz)[:-4]
        if base.endswith("_maps"):
            continue
        z = np.load(pnpz)
        y = z["y_true"]
        for r in [x for x in per_fold
                  if f"{x['protocol']}_f{x['fold']}" == base]:
            m = r["model"]
            key = f"scores_{m}"
            if key not in z:
                mism.append(f"{base}: predictions missing {key}")
                continue
            s = z[key]
            th = float(z[f"theta_{m}"])
            yp = (s >= th).astype(int)
            got_f1 = f1_score(y, yp, average="macro", zero_division=0) * 100
            got_ac = accuracy_score(y, yp) * 100
            exp = r["at_theta_star"]
            if abs(got_f1 - exp["macro_f1"]) > 1e-6:
                mism.append(f"{base}/{m}: macro F1 stored {exp['macro_f1']:.6f} "
                            f"vs recomputed {got_f1:.6f}")
            if abs(got_ac - exp["accuracy"]) > 1e-6:
                mism.append(f"{base}/{m}: accuracy stored {exp['accuracy']:.6f} "
                            f"vs recomputed {got_ac:.6f}")
            if abs(th - exp["threshold"]) > 1e-9:
                mism.append(f"{base}/{m}: threshold stored {exp['threshold']} vs {th}")
            if "roc_auc" in exp and y.sum() not in (0, len(y)):
                got_auc = roc_auc_score(y, s) * 100
                if abs(got_auc - exp["roc_auc"]) > 1e-6:
                    mism.append(f"{base}/{m}: AUC stored {exp['roc_auc']:.6f} "
                                f"vs recomputed {got_auc:.6f}")
            n_re += 1
    if mism:
        ck.fail("stored metrics reproduce from predictions", "\n        ".join(mism[:20]))
    else:
        ck.ok("stored metrics reproduce from predictions",
              f"{n_re} model/fold combinations recomputed exactly")

    # ---- 6. aggregate benchmark equals the per-fold mean -----------------
    bp = os.path.join(run, "metrics", "benchmark.json")
    if os.path.exists(bp):
        bench = json.load(open(bp))
        agg_bad = []
        for proto, models in bench.items():
            for m, b in models.items():
                rows = [r for r in per_fold if r["protocol"] == proto and r["model"] == m]
                if not rows:
                    agg_bad.append(f"{proto}/{m}: aggregate exists but no per-fold rows")
                    continue
                exp = float(np.mean([r["at_theta_star"]["macro_f1"] for r in rows]))
                if abs(exp - b["macro_f1_mean"]) > 1e-9:
                    agg_bad.append(f"{proto}/{m}: mean stored {b['macro_f1_mean']} vs {exp}")
                if b["n_folds"] != len(rows):
                    agg_bad.append(f"{proto}/{m}: n_folds {b['n_folds']} vs {len(rows)}")
        if agg_bad:
            ck.fail("aggregate equals per-fold mean", "\n        ".join(agg_bad[:20]))
        else:
            ck.ok("aggregate equals per-fold mean")

    # ---- 7. numbers.tex and manuscript literals (Rule 7) ----------------
    ntex = os.path.join(ROOT, "paper", "numbers.tex")
    macros = {}
    if os.path.exists(ntex):
        for mm in re.finditer(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}", open(ntex).read()):
            macros[mm.group(1)] = mm.group(2)
        ck.ok("paper/numbers.tex exists", f"{len(macros)} macros")
    else:
        ck.fail("paper/numbers.tex exists", "run visualize.py")

    pool = collect_json_numbers(run)
    # config values are a legitimate source for structural facts (state dimension,
    # sample counts, hyperparameters), not only the metrics files
    pool |= collect_json_numbers_from_file(os.path.join(ROOT, "configs", "config.json"))
    if macros:
        orphan_macros = []
        for k, v in macros.items():
            try:
                x = float(v)
            except ValueError:
                continue
            # tolerance must cover the printed precision: a macro formatted with
            # {:.3g} legitimately differs from the stored value by up to half a
            # unit in its last printed digit.
            digits = len(v.split(".")[1]) if "." in v else 0
            tol = max(0.5 * 10 ** (-digits), 5e-4 * abs(x))
            if not close_to_any(x, pool, tol=tol):
                orphan_macros.append(f"\\{k} = {v} appears in no metrics JSON or config")
        if orphan_macros:
            ck.fail("every macro traces to a metrics JSON", "\n        ".join(orphan_macros))
        else:
            ck.ok("every macro traces to a metrics JSON",
                  f"{len(macros)} macros checked against {len(pool)} stored values")

    mtex = os.path.join(ROOT, "paper", "manuscript.tex")
    if os.path.exists(mtex):
        body = strip_structural(open(mtex).read())
        lits = set()
        for mm in re.finditer(r"(?<![\w.\\])(\d+\.\d+)(?![\d])", body):
            lits.add(mm.group(1))
        allow = set()
        cfgp = os.path.join(ROOT, "configs", "config.json")
        cfg_pool = collect_json_numbers_from_file(cfgp)
        orphans = []
        for s in sorted(lits):
            x = float(s)
            if close_to_any(x, cfg_pool, tol=1e-9):
                continue
            if 1900 <= x <= 2100:
                continue
            orphans.append(s)
        if orphans:
            ck.fail("manuscript prints no typed numeric result",
                    "typed literals with no config source: " + ", ".join(orphans) +
                    "\n        -> replace each with a macro from numbers.tex")
        else:
            ck.ok("manuscript prints no typed numeric result")

        # every macro the manuscript uses must be defined in numbers.tex
        used = set(re.findall(r"\\([a-zA-Z]{4,})", open(mtex).read()))
        latex_builtin = {
            "documentclass", "usepackage", "begin", "end", "title", "author", "thanks",
            "maketitle", "section", "subsection", "subsubsection", "label", "ref",
            "eqref", "cite", "input", "includegraphics", "caption", "centering",
            "textbf", "textit", "emph", "item", "today", "mathbf", "mathrm", "top",
            "frac", "sqrt", "nabla", "partial", "times", "pm", "ge", "le", "approx",
            "left", "right", "toprule", "midrule", "bottomrule", "multicolumn",
            "newcommand", "linewidth", "textwidth", "bigoplus", "mathbb", "quad",
            "hidelinks", "url", "sigma", "delta", "theta", "alpha", "mu", "hat",
            "top", "leftarrow", "rightarrow", "text", "colon", "dots", "cdots",
            "IEEEPARstart", "IEEEkeywords", "bfseries", "itshape", "footnotesize",
            "small", "large", "clearpage", "newpage", "noindent", "par", "vspace",
            "hspace", "setlength", "textsuperscript", "texttt", "mathcal",
            "Delta", "articletype", "affil", "keywords", "fancyhead", "orcid",
            "email", "href", "tabular", "oldtabular", "newcolumntype",
            "arraybackslash", "graphicspath", "makeatletter", "makeatother",
            "reset", "color", "FloatBarrier", "bibliography", "bibliographystyle",
            "appendix", "enumerate", "itemize", "hline", "cline",
            "cdot", "fontsize", "ldots", "normalsize", "omega", "qquad",
            "raggedright", "renewcommand", "selectfont", "widehat", "hypersetup",
        }
        undefined = sorted(u for u in used
                           if u not in macros and u not in latex_builtin)
        if undefined:
            ck.fail("every macro used in the manuscript is defined",
                    "undefined: " + ", ".join(undefined) +
                    "\n        -> LaTeX would fail, or silently print nothing")
        else:
            ck.ok("every macro used in the manuscript is defined",
                  f"{len(used - latex_builtin)} project macros resolved")

        # tables must be \input, not inline tabulars
        raw = open(mtex).read()
        n_inputs = len(re.findall(r"\\input\{[^}]*table[^}]*\}", raw))
        n_inline = len(re.findall(r"\\begin\{tabular\}", raw))
        if n_inline > 0:
            ck.fail("tables are generated, not inline",
                    f"{n_inline} inline tabular environment(s) in manuscript.tex")
        else:
            ck.ok("tables are generated, not inline", f"{n_inputs} table inputs")

        # every referenced figure exists
        missing = []
        for mm in re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", raw):
            rel = mm.group(1)
            cand = [os.path.join(ROOT, "paper", rel), os.path.join(ROOT, rel)]
            if not any(os.path.exists(c) or os.path.exists(c + ".png") for c in cand):
                missing.append(rel)
        if missing:
            ck.fail("every referenced figure exists", ", ".join(missing))
        else:
            ck.ok("every referenced figure exists")

    # ---- 8. generator scripts survive -----------------------------------
    need = ["dataset.py", "train.py", "evaluate.py", "visualize.py", "verify.py"]
    miss = [n for n in need if not os.path.exists(os.path.join(ROOT, n))]
    if miss:
        ck.fail("all generator scripts present", f"missing: {miss}")
    else:
        ck.ok("all generator scripts present", ", ".join(need))

    # ---- 9. explicit not-measured record --------------------------------
    nmp = os.path.join(run, "metrics", "not_measured.json")
    if not os.path.exists(nmp):
        ck.fail("not_measured.json exists",
                "Rule 9 requires unmeasured quantities to be recorded explicitly")
    else:
        nm = json.load(open(nmp))
        unmeasured = [k for k, v in nm.items() if not v.get("measured", True)]
        ck.ok("not_measured.json exists",
              f"{len(unmeasured)} quantities declared not measurable: {unmeasured}")

    return ck.report()


def collect_json_numbers_from_file(path):
    vals = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            vals.add(float(o))

    if os.path.exists(path):
        walk(json.load(open(path)))
    return vals


if __name__ == "__main__":
    sys.exit(main())
