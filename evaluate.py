"""
evaluate.py -- load checkpoints, run inference, compute EVERY reported number.

Prohibitions (01_system.md Rule 2): must not train or refit anything, must not
plot. Every array any figure will need is computed and saved here, because
visualize.py is forbidden to compute (Rule 3).

All metrics are written at full float precision. Nothing is rounded here.

Usage:  python evaluate.py [--run RUN_ID]
"""

import argparse
import json
import os
import time

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)

import dataset as D
import models as M

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ORDER = ["peak1", "pca10", "esn50", "esn500", "rc500", "patch4500", "patch4500_filt"]
PRETTY = {
    "peak1": "Peak amplitude + linear (1D)",
    "pca10": "PCA-10 + linear (10D)",
    "esn50": "Software ESN, 50 neurons (50D)",
    "esn500": "Software ESN, 500 neurons (500D)",
    "rc500": "Physical RC, raw state (500D)",
    "patch4500": "Physical RC, 3x3 patch state (4500D)",
    "patch4500_filt": "Physical RC, patch + spatial filter (4500D)",
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def classification_metrics(y, scores, threshold):
    yp = (scores >= threshold).astype(int)
    cm = confusion_matrix(y, yp, labels=[0, 1])
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, yp) * 100),
        "macro_f1": float(f1_score(y, yp, average="macro", zero_division=0) * 100),
        "macro_precision": float(precision_score(y, yp, average="macro", zero_division=0) * 100),
        "macro_recall": float(recall_score(y, yp, average="macro", zero_division=0) * 100),
        "defect_precision": float(precision_score(y, yp, pos_label=1, zero_division=0) * 100),
        "defect_recall": float(recall_score(y, yp, pos_label=1, zero_division=0) * 100),
        "defect_f1": float(f1_score(y, yp, pos_label=1, zero_division=0) * 100),
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y)),
        "n_test_positive": int(y.sum()),
    }
    if y.sum() > 0 and y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, scores) * 100)
        p = sigmoid(scores)
        out["bce_log_loss"] = float(log_loss(y, p, labels=[0, 1]))
        out["brier_score"] = float(brier_score_loss(y, p))
        ece, mce, curve = reliability(y, p)
        out["ece_percent"] = ece
        out["mce_percent"] = mce
        out["reliability_curve"] = curve
    return out


def reliability(y, p, n_bins=10):
    """Measured calibration curve. These are the ONLY numbers a reliability
    diagram may plot -- never an analytic stand-in (01_system.md Rule 4)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    curve = {"bin_lo": [], "bin_hi": [], "n": [], "mean_confidence": [], "observed_frequency": []}
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < n_bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        curve["bin_lo"].append(float(edges[i]))
        curve["bin_hi"].append(float(edges[i + 1]))
        curve["n"].append(int(m.sum()))
        if m.sum() == 0:
            curve["mean_confidence"].append(None)
            curve["observed_frequency"].append(None)
            continue
        conf, obs = float(p[m].mean()), float(y[m].mean())
        curve["mean_confidence"].append(conf)
        curve["observed_frequency"].append(obs)
        gap = abs(conf - obs)
        ece += m.sum() / len(y) * gap
        mce = max(mce, gap)
    return float(ece * 100), float(mce * 100), curve


def score_all_models(ck, ds, run, tag, cfg):
    """Return {model_name: scores over the whole grid (flat, serpentine-free order)}."""
    T = ds["T"]
    sY, sX = ds["sY"], ds["sX"]
    mc = cfg["models"]
    Xn = (ds["X_grid"].reshape(-1, T) - ck["mean"]) / ck["std"]
    scores = {}
    maps = {}

    for name in ["peak1", "pca10", "rc500"]:
        m = ck["models"][name]
        scores[name] = m["clf"].decision_function(m["feature"].transform(Xn))

    for size in mc["esn"]["sizes"]:
        name = f"esn{size}"
        H = np.load(os.path.join(run, "states", f"{tag}_esn{size}.npy"))
        scores[name] = ck["models"][name]["clf"].decision_function(H)
        del H

    Xdiff = ((ds["X_grid"] - ck["ground_mean"].reshape(1, 1, T)) / ck["std"].reshape(1, 1, T)
             ).astype(np.float32)
    Pf = D.build_patch_features(Xdiff, mc["patch_size"]).reshape(-1, mc["patch_size"] ** 2 * T)
    s_patch = ck["models"]["patch4500"]["clf"].decision_function(Pf)
    scores["patch4500"] = s_patch
    pp = cfg["postprocess"]
    raw_map = s_patch.reshape(sY, sX)
    filt_map = gaussian_filter(median_filter(raw_map, size=pp["median_size"]),
                               sigma=pp["gaussian_sigma"])
    scores["patch4500_filt"] = filt_map.ravel()
    maps["patch_raw"] = raw_map
    maps["patch_filtered"] = filt_map
    maps["rc500"] = scores["rc500"].reshape(sY, sX)
    del Pf, Xdiff
    return scores, maps


def build_derived(run, cfg, per_fold, agg, T):
    """Compute derived summary quantities from per_fold.json, benchmark.json, and predictions."""
    diam_mm = cfg["dataset"]["defect_diameter_mm"]
    groups = {}
    for i, d in enumerate(diam_mm, start=1):
        groups.setdefault(f"diameter_{d:g}mm", []).append(i)

    derived = {"by_defect_diameter": {}, "protocol_gap_macro_f1": {},
               "note": ("Aggregates and differences over values already stored in "
                        "per_fold.json / benchmark.json. Computed here so that "
                        "visualize.py only reads.")}
    for name in MODEL_ORDER:
        rows = [r for r in per_fold if r["protocol"] == "loco_guard" and r["model"] == name]
        if not rows:
            continue
        g = {}
        for gname, ids in groups.items():
            vals = [r["at_theta_star"]["macro_f1"] for r in rows
                    if r["held_out_defect"] in ids]
            rec = [r["at_theta_star"]["defect_recall"] for r in rows
                   if r["held_out_defect"] in ids]
            if vals:
                g[gname] = {"n_folds": len(vals),
                            "macro_f1_mean": float(np.mean(vals)),
                            "macro_f1_std": float(np.std(vals)),
                            "defect_recall_mean": float(np.mean(rec))}
        derived["by_defect_diameter"][name] = g

        gaps = {}
        for a, b in [("random_optimistic", "loco_guard"),
                     ("random_optimistic", "blocked"),
                     ("blocked", "loco_guard")]:
            if name in agg.get(a, {}) and name in agg.get(b, {}):
                gaps[f"{a}_minus_{b}"] = float(agg[a][name]["macro_f1_mean"]
                                               - agg[b][name]["macro_f1_mean"])
        derived["protocol_gap_macro_f1"][name] = gaps

    # effect of spatial post-filtering, per protocol (the mechanism claim)
    derived["spatial_filter_effect_macro_f1"] = {}
    for proto in agg:
        if "patch4500" in agg[proto] and "patch4500_filt" in agg[proto]:
            derived["spatial_filter_effect_macro_f1"][proto] = float(
                agg[proto]["patch4500_filt"]["macro_f1_mean"]
                - agg[proto]["patch4500"]["macro_f1_mean"])

    # calibration summary (ECE, Brier, and pooled LOCO reliability bins)
    derived["calibration_summary"] = {}
    for proto in ["loco_guard", "blocked", "random_optimistic"]:
        derived["calibration_summary"][proto] = {}
        for m in ["rc500", "patch4500_filt"]:
            m_rows = [r for r in per_fold if r["protocol"] == proto and r["model"] == m
                      and "reliability_curve" in r.get("at_theta_star", {})]
            if not m_rows:
                continue
            ece_mean = float(np.mean([r["at_theta_star"]["ece_percent"] for r in m_rows]))
            brier_mean = float(np.mean([r["at_theta_star"]["brier_score"] for r in m_rows]))
            total_n = np.zeros(10)
            total_pos = np.zeros(10)
            for r in m_rows:
                c = r["at_theta_star"]["reliability_curve"]
                for i, (n_val, f_val) in enumerate(zip(c["n"], c["observed_frequency"])):
                    if n_val > 0 and f_val is not None:
                        total_n[i] += n_val
                        total_pos[i] += f_val * n_val
            obs_pct = [float(100.0 * total_pos[i] / total_n[i]) if total_n[i] > 0 else None
                       for i in range(10)]
            derived["calibration_summary"][proto][m] = {
                "ece_mean_percent": ece_mean,
                "brier_mean": brier_mean,
                "pooled_observed_frequency_percent": obs_pct,
            }

    # Fold 4 score diagnostics (from stored predictions/loco_guard_f4.npz)
    f4_path = os.path.join(run, "predictions", "loco_guard_f4.npz")
    if os.path.exists(f4_path):
        z4 = np.load(f4_path)
        y4 = z4["y_true"]
        s4 = z4["scores_patch4500_filt"]
        th4 = float(z4["theta_patch4500_filt"])
        derived["fold4_diagnostics"] = {
            "defect_score_min": float(s4[y4 == 1].min()),
            "defect_score_max": float(s4[y4 == 1].max()),
            "defect_score_mean": float(s4[y4 == 1].mean()),
            "background_score_mean": float(s4[y4 == 0].mean()),
            "threshold": th4,
            "gap_below_threshold": float(th4 - s4[y4 == 1].max()),
        }

    # dataset facts the paper quotes, resolved to a metrics file
    derived["dataset"] = {
        "n_scan_points": int(cfg["dataset"]["expected_n_points"]),
        "n_defect_pixels": int(cfg["dataset"]["expected_n_positive"]),
        "n_defects": int(len(cfg["dataset"]["defect_circles_xyr"])),
        "defect_fraction_percent": float(100.0 * cfg["dataset"]["expected_n_positive"]
                                         / cfg["dataset"]["expected_n_points"]),
        "state_dim": int(T),
        "patch_dim": int(cfg["models"]["patch_size"] ** 2 * T),
        "variance_thresholds_percent": [90.0, 95.0, 99.0, 99.9],
    }
    with open(os.path.join(run, "metrics", "derived.json"), "w") as f:
        json.dump(derived, f, indent=2)
    return derived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--refresh-derived", action="store_true")
    ap.add_argument("--update-extra", action="store_true")
    ap.add_argument("--fix-esn-noise", action="store_true")
    args = ap.parse_args()

    cfg = D.load_config()
    run_id = args.run or open(os.path.join(ROOT, "runs", "LATEST")).read().strip()
    run = os.path.join(ROOT, "runs", run_id)
    if args.refresh_derived:
        with open(os.path.join(run, "metrics", "per_fold.json")) as f:
            per_fold = json.load(f)
        with open(os.path.join(run, "metrics", "benchmark.json")) as f:
            agg = json.load(f)
        build_derived(run, cfg, per_fold, agg, cfg["dataset"]["transient_length"])
        log(f"refreshed derived.json for run {run_id}")
        return
    if args.fix_esn_noise:
        ds = D.load_dataset(cfg, validate=False)
        fix_esn_noise_and_welch(run, ds, cfg)
        with open(os.path.join(run, "metrics", "per_fold.json")) as f:
            per_fold = json.load(f)
        with open(os.path.join(run, "metrics", "benchmark.json")) as f:
            agg = json.load(f)
        build_derived(run, cfg, per_fold, agg, cfg["dataset"]["transient_length"])
        log(f"fixed esn noise and welch metadata for run {run_id}")
        return
    if args.update_extra:
        ds = D.load_dataset(cfg, validate=False)
        update_extra_metrics(run, ds, cfg)
        with open(os.path.join(run, "metrics", "per_fold.json")) as f:
            per_fold = json.load(f)
        with open(os.path.join(run, "metrics", "benchmark.json")) as f:
            agg = json.load(f)
        build_derived(run, cfg, per_fold, agg, cfg["dataset"]["transient_length"])
        log(f"updated extra metrics and derived.json for run {run_id}")
        return
    log(f"evaluating run {run_id}")

    ds = D.load_dataset(cfg, validate=False)
    sY, sX, T = ds["sY"], ds["sX"], ds["T"]
    y_all = ds["label_map"].ravel()

    results = {}
    per_fold = []
    sweeps = {}
    loco_rc_bg = np.zeros((sY, sX), dtype=float)
    loco_pr_bg = np.zeros((sY, sX), dtype=float)
    loco_pf_bg = np.zeros((sY, sX), dtype=float)
    loco_bg_cnt = np.zeros((sY, sX), dtype=float)
    loco_rc_oof = np.zeros((sY, sX), dtype=float)
    loco_pr_oof = np.zeros((sY, sX), dtype=float)
    loco_pf_oof = np.zeros((sY, sX), dtype=float)
    loco_in_test = np.zeros((sY, sX), dtype=bool)
    loco_th_rc = []
    loco_th_pf = []

    cks = sorted(os.listdir(os.path.join(run, "checkpoints")))
    for ci, fn in enumerate(cks, 1):
        tag = fn.replace(".joblib", "")
        ck = joblib.load(os.path.join(run, "checkpoints", fn))
        mk = np.load(os.path.join(run, "masks", f"{tag}.npz"))
        i_te = np.flatnonzero(mk["test_mask"].ravel())
        i_val = np.flatnonzero(mk["val_mask"].ravel())
        log(f"[{ci}/{len(cks)}] {tag}: scoring {len(i_te)} test pixels")

        scores, maps = score_all_models(ck, ds, run, tag, cfg)

        for name in MODEL_ORDER:
            s = scores[name]
            th = ck["models"][name]["threshold_val"]
            rec = {
                "protocol": ck["protocol"], "fold": ck["fold"],
                "held_out_defect": ck["held_out_defect"], "model": name,
                "model_pretty": PRETTY[name],
                "state_dim": ck["models"][name]["state_dim"],
                "val_macro_f1": ck["models"][name]["val_macro_f1"],
                "at_theta_star": classification_metrics(y_all[i_te], s[i_te], th),
                "at_theta_zero": classification_metrics(y_all[i_te], s[i_te], 0.0),
            }
            per_fold.append(rec)

        np.savez_compressed(
            os.path.join(run, "predictions", f"{tag}.npz"),
            test_index=i_te, y_true=y_all[i_te],
            **{f"scores_{n}": scores[n][i_te] for n in MODEL_ORDER},
            **{f"theta_{n}": ck["models"][n]["threshold_val"] for n in MODEL_ORDER},
        )

        # threshold sweep curve on VALIDATION, for the calibration figure
        if ck["protocol"] in ("loco_guard",) and ck["fold"] == 1 or ck["protocol"] != "loco_guard":
            ts = cfg["threshold_sweep"]
            sw = {}
            for name in ["rc500", "patch4500", "patch4500_filt"]:
                _, _, ths, f1c = M.sweep_threshold(y_all[i_val], scores[name][i_val],
                                                   ts["min"], ts["max"], ts["num_steps"])
                sw[name] = {"thresholds": ths.tolist(), "val_macro_f1": (f1c * 100).tolist(),
                            "selected": ck["models"][name]["threshold_val"]}
            sweeps[tag] = sw

        # 2-D maps + ROC, for the representative protocols only (keeps files small)
        if ck["protocol"] == "loco_guard":
            te_m = mk["test_mask"]
            loco_rc_oof[te_m] = maps["rc500"][te_m]
            loco_pr_oof[te_m] = maps["patch_raw"][te_m]
            loco_pf_oof[te_m] = maps["patch_filtered"][te_m]
            loco_in_test |= te_m
            loco_rc_bg[~te_m] += maps["rc500"][~te_m]
            loco_pr_bg[~te_m] += maps["patch_raw"][~te_m]
            loco_pf_bg[~te_m] += maps["patch_filtered"][~te_m]
            loco_bg_cnt[~te_m] += 1.0
            loco_th_rc.append(ck["models"]["rc500"]["threshold_val"])
            loco_th_pf.append(ck["models"]["patch4500_filt"]["threshold_val"])
        if ck["protocol"] in ("blocked", "random_optimistic"):
            np.savez_compressed(
                os.path.join(run, "predictions", f"{tag}_maps.npz"),
                label_map=ds["label_map"], test_mask=mk["test_mask"],
                map_rc500=maps["rc500"], map_patch_raw=maps["patch_raw"],
                map_patch_filtered=maps["patch_filtered"],
                theta_rc500=ck["models"]["rc500"]["threshold_val"],
                theta_patch_filt=ck["models"]["patch4500_filt"]["threshold_val"])
            roc = {}
            for name in MODEL_ORDER:
                fpr, tpr, _ = roc_curve(y_all[i_te], scores[name][i_te])
                keep = np.linspace(0, len(fpr) - 1, min(len(fpr), 2000)).astype(int)
                roc[name] = {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist(),
                             "auc": float(roc_auc_score(y_all[i_te], scores[name][i_te]) * 100),
                             "pretty": PRETTY[name]}
            with open(os.path.join(run, "metrics", f"roc_{ck['protocol']}.json"), "w") as f:
                json.dump(roc, f, indent=2)

        # ---- noise robustness, on the blocked protocol ----------------------
        if ck["protocol"] == "blocked":
            log("    noise robustness sweep ...")
            ne = cfg["noise_evaluation"]
            rng = np.random.RandomState(cfg["protocols"]["random_seed"])
            Xn0 = (ds["X_grid"].reshape(-1, T) - ck["mean"]) / ck["std"]
            sig_pow = float(np.mean(Xn0 ** 2))
            noise = {"snr_db": ne["snr_db"], "n_repeats": ne["n_repeats"], "models": {}}
            targets = ne.get("models", ["rc500", "esn50", "patch4500_filt"])
            for n in targets:
                noise["models"][n] = {"macro_f1_mean": [], "macro_f1_std": [],
                                      "accuracy_mean": [], "pretty": PRETTY[n]}
            for snr in ne["snr_db"]:
                sd = np.sqrt(sig_pow / (10.0 ** (snr / 10.0)))
                acc_r = {n: [] for n in targets}
                f1_r = {n: [] for n in targets}
                for _ in range(ne["n_repeats"]):
                    Xnz = Xn0 + rng.normal(0, sd, size=Xn0.shape)
                    for n in targets:
                        if n == "rc500":
                            s = ck["models"]["rc500"]["clf"].decision_function(Xnz)
                        elif n.startswith("esn"):
                            s = ck["models"][n]["clf"].decision_function(
                                ck["models"][n]["esn"].states(Xnz))
                        else:
                            Xd = ((Xnz * ck["std"] + ck["mean"] - ck["ground_mean"])
                                  / ck["std"]).reshape(sY, sX, T).astype(np.float32)
                            Pf = D.build_patch_features(Xd, cfg["models"]["patch_size"]
                                                        ).reshape(-1, 9 * T)
                            sm = ck["models"]["patch4500"]["clf"].decision_function(Pf
                                                                                   ).reshape(sY, sX)
                            pp = cfg["postprocess"]
                            s = gaussian_filter(median_filter(sm, size=pp["median_size"]),
                                                sigma=pp["gaussian_sigma"]).ravel()
                            del Pf, Xd
                        th = ck["models"][n]["threshold_val"]
                        yp = (s[i_te] >= th).astype(int)
                        f1_r[n].append(f1_score(y_all[i_te], yp, average="macro", zero_division=0) * 100)
                        acc_r[n].append(accuracy_score(y_all[i_te], yp) * 100)
                    del Xnz
                for n in targets:
                    noise["models"][n]["macro_f1_mean"].append(float(np.mean(f1_r[n])))
                    noise["models"][n]["macro_f1_std"].append(float(np.std(f1_r[n])))
                    noise["models"][n]["accuracy_mean"].append(float(np.mean(acc_r[n])))
                log(f"      SNR {snr:>5} dB done")
            with open(os.path.join(run, "metrics", "noise_robustness.json"), "w") as f:
                json.dump(noise, f, indent=2)

            # ---- reservoir characterisation, on this fold's FIT split -------
            log("    reservoir characterisation ...")
            i_fit = np.flatnonzero(mk["fit_mask"].ravel())
            res = characterise(Xn0[i_fit], y_all[i_fit], cfg)
            res["temporal_windows"] = temporal_windows(Xn0, y_all, i_fit,
                                                       np.flatnonzero(mk["test_mask"].ravel()), cfg)
            with open(os.path.join(run, "metrics", "reservoir_characterisation.json"), "w") as f:
                json.dump(res, f, indent=2)

        # ---- LOCO diameter regression -------------------------------------
        if ck["protocol"] == "loco_guard" and "diam_rc500" in ck["models"]:
            did = ds["defect_id_map"].ravel()
            te_def = i_te[did[i_te] > 0]
            if len(te_def) > 0:
                pred = ck["models"]["diam_rc500"]["reg"].predict(
                    ((ds["X_grid"].reshape(-1, T)[te_def] - ck["mean"]) / ck["std"]))
                results.setdefault("diameter_loco", []).append({
                    "fold": ck["fold"], "held_out_defect": ck["held_out_defect"],
                    "true_diameter_mm": float(ds["diameter_mm"][ck["held_out_defect"] - 1]),
                    "pred_diameter_mm_mean": float(pred.mean()),
                    "pred_diameter_mm_std": float(pred.std()),
                    "n_defect_pixels": int(len(te_def))})
        del scores, maps

    # ------------------------------------------------------------------ #
    if loco_th_pf:
        loco_rc_oof[~loco_in_test] = loco_rc_bg[~loco_in_test] / np.maximum(loco_bg_cnt[~loco_in_test], 1.0)
        loco_pr_oof[~loco_in_test] = loco_pr_bg[~loco_in_test] / np.maximum(loco_bg_cnt[~loco_in_test], 1.0)
        loco_pf_oof[~loco_in_test] = loco_pf_bg[~loco_in_test] / np.maximum(loco_bg_cnt[~loco_in_test], 1.0)
        np.savez_compressed(
            os.path.join(run, "predictions", "loco_guard_oof_maps.npz"),
            label_map=ds["label_map"], test_mask=loco_in_test,
            map_rc500=loco_rc_oof, map_patch_raw=loco_pr_oof,
            map_patch_filtered=loco_pf_oof,
            theta_rc500=float(np.mean(loco_th_rc)),
            theta_patch_filt=float(np.mean(loco_th_pf)))
    with open(os.path.join(run, "metrics", "per_fold.json"), "w") as f:
        json.dump(per_fold, f, indent=2)
    with open(os.path.join(run, "metrics", "threshold_sweeps.json"), "w") as f:
        json.dump(sweeps, f, indent=2)

    # aggregate per protocol
    agg = {}
    for proto in ["loco_guard", "blocked", "random_optimistic"]:
        agg[proto] = {}
        for name in MODEL_ORDER:
            rows = [r for r in per_fold if r["protocol"] == proto and r["model"] == name]
            if not rows:
                continue
            g = lambda k: np.array([r["at_theta_star"][k] for r in rows], dtype=float)
            agg[proto][name] = {
                "model_pretty": PRETTY[name],
                "state_dim": rows[0]["state_dim"],
                "n_folds": len(rows),
                "accuracy_mean": float(g("accuracy").mean()),
                "accuracy_std": float(g("accuracy").std()),
                "macro_f1_mean": float(g("macro_f1").mean()),
                "macro_f1_std": float(g("macro_f1").std()),
                "macro_precision_mean": float(g("macro_precision").mean()),
                "macro_recall_mean": float(g("macro_recall").mean()),
                "defect_recall_mean": float(g("defect_recall").mean()),
                "defect_precision_mean": float(g("defect_precision").mean()),
                "defect_f1_mean": float(g("defect_f1").mean()),
                "roc_auc_mean": float(g("roc_auc").mean()) if "roc_auc" in rows[0]["at_theta_star"] else None,
                "threshold_mean": float(g("threshold").mean()),
            }
    with open(os.path.join(run, "metrics", "benchmark.json"), "w") as f:
        json.dump(agg, f, indent=2)

    # diameter regression summary
    if "diameter_loco" in results:
        rows = results["diameter_loco"]
        t = np.array([r["true_diameter_mm"] for r in rows])
        p = np.array([r["pred_diameter_mm_mean"] for r in rows])
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        summary = {
            "per_fold": rows,
            "n_folds": len(rows),
            "r2_leave_one_defect_out": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
            "rmse_mm": float(np.sqrt(((t - p) ** 2).mean())),
            "mae_mm": float(np.abs(t - p).mean()),
            "caveat": ("Only 9 defects exist and only 3 distinct diameters (10/8/6 mm). "
                       "Under leave-one-defect-out there are 8 training defects and at most "
                       "2 examples of the held-out diameter. R^2 here is computed over 9 "
                       "points and should not be presented as a continuous sizing capability."),
        }
        with open(os.path.join(run, "metrics", "diameter_regression.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # ---- derived quantities the paper quotes (Rule 2: computed HERE, not in
    # visualize.py, which is forbidden to compute anything at all) -----------
    build_derived(run, cfg, per_fold, agg, T)

    # things we explicitly did NOT measure (01_system.md Rule 9)
    notmeas = {
        "memory_capacity": {
            "measured": False,
            "reason": ("Memory capacity in the sense of Jaeger (2001) requires driving the "
                       "reservoir with a temporally extended input stream and training "
                       "readouts to reconstruct u(t-k) from the reservoir state. This dataset "
                       "contains one independent Gaussian pulse response per scan location, "
                       "with no sequential excitation, so MC is not identifiable. idea.md "
                       "anticipated exactly this restriction."),
            "do_not_substitute": ("The lagged autocorrelation of a scan-averaged waveform is "
                                  "NOT memory capacity; it measures smoothness and any "
                                  "low-pass signal scores highly on it."),
        },
        "defect_depth_and_volume": {
            "measured": False,
            "reason": ("The project's own sources disagree on the ground-truth depths "
                       "(dataset_loader.py lists 0.86/0.66/0.48... mm; the previous manuscript "
                       "Table 2B lists 1/2/3 mm). Without a documented depth reference, depth "
                       "and volume regression cannot be evaluated honestly."),
        },
        "physical_size_calibration": {
            "measured": False,
            "reason": ("The pixel-to-millimetre scale (1 px = 1 mm) is implied by the project's "
                       "own claim that r=5 px corresponds to d=10 mm, but is not independently "
                       "documented. All millimetre statements inherit this assumption."),
        },
        "inference_time": {
            "measured": True,
            "note": "See timing.json; measured with a timer, reported per scan point.",
        },
    }
    with open(os.path.join(run, "metrics", "not_measured.json"), "w") as f:
        json.dump(notmeas, f, indent=2)

    log(f"EVALUATION COMPLETE -> runs/{run_id}/metrics")


# --------------------------------------------------------------------------- #
def _spec_and_fisher(Z, y, max_ev=80, max_fisher_pcs=50, total_var_override=None):
    """Compute eigenvalue spectrum, effective dimensionality, and per-PC Fisher ratios."""
    Zc = Z.astype(np.float64) - Z.mean(axis=0, keepdims=True)
    n, d = Zc.shape
    C = (Zc.T @ Zc) / (n - 1)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    ev = np.maximum(vals[idx], 0.0)
    V = vecs[:, idx[:min(max_fisher_pcs, d)]]
    tot = float(total_var_override) if total_var_override is not None else float(ev.sum())
    ratio = ev / (tot + 1e-300)
    cum = np.cumsum(ratio)
    pi = ratio[ratio > 1e-300]
    pr = float((tot ** 2) / (np.sum(ev ** 2) + 1e-300))
    d_se = float(np.exp(-np.sum(pi * np.log(pi))))

    P = Zc @ V
    P0, P1 = P[y == 0], P[y == 1]
    mu0, mu1 = P0.mean(axis=0), P1.mean(axis=0)
    v0, v1 = P0.var(axis=0), P1.var(axis=0)
    pc_fisher = ((mu1 - mu0) ** 2 / (v0 + v1 + 1e-12)).astype(float)
    cum_fisher = np.cumsum(pc_fisher)

    return {
        "state_dim": int(d if total_var_override is None else 4500),
        "eigenvalues": ev[:max_ev].tolist(),
        "norm_eigenvalues": ratio[:max_ev].tolist(),
        "cumulative_variance": cum[:min(320, len(cum))].tolist(),
        "participation_ratio": pr,
        "spectral_entropy_dimension": d_se,
        "n_pcs_90": int(np.argmax(cum >= 0.90) + 1) if np.any(cum >= 0.90) else int(len(cum)),
        "n_pcs_95": int(np.argmax(cum >= 0.95) + 1) if np.any(cum >= 0.95) else int(len(cum)),
        "n_pcs_99": int(np.argmax(cum >= 0.99) + 1) if np.any(cum >= 0.99) else int(len(cum)),
        "pc_fisher_ratio": pc_fisher.tolist(),
        "cumulative_pc_fisher": cum_fisher.tolist(),
        "total_pc_fisher_top50": float(cum_fisher[-1]) if len(cum_fisher) > 0 else 0.0,
    }


def characterise(X, y, cfg, run=None, ds=None, ck=None, i_fit=None):
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    vals_full, vecs_full = np.linalg.eigh(C)
    idx_full = np.argsort(vals_full)[::-1]
    ev = np.maximum(vals_full[idx_full], 0.0)
    tot = ev.sum()
    ratio = ev / (tot + 1e-300)
    cum = np.cumsum(ratio)
    pi = ratio[ratio > 1e-300]
    S = np.sqrt((len(Xc) - 1) * ev)
    out = {
        "n_samples": int(len(X)), "state_dim": int(X.shape[1]),
        "eigenvalues": ev.tolist(),
        "cumulative_variance": cum.tolist(),
        "participation_ratio": float(tot ** 2 / (np.sum(ev ** 2) + 1e-300)),
        "spectral_entropy_dimension": float(np.exp(-np.sum(pi * np.log(pi)))),
        "n_pcs_90": int(np.argmax(cum >= 0.90) + 1),
        "n_pcs_95": int(np.argmax(cum >= 0.95) + 1),
        "n_pcs_99": int(np.argmax(cum >= 0.99) + 1),
        "n_pcs_999": int(np.argmax(cum >= 0.999) + 1),
        "condition_number": float(S[0] / (S[-1] + 1e-300)),
        "interpretation_note": ("Participation ratio and spectral-entropy dimension describe "
                               "the effective dimensionality; the 99.9%-variance count is "
                               "dominated by the noise floor of a smoothly decaying spectrum "
                               "and must not be quoted alone as evidence of high-dimensional "
                               "expansion. None of these quantities measures non-linearity."),
    }
    mu0, mu1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)
    v0, v1 = X[y == 0].var(axis=0), X[y == 1].var(axis=0)
    out["max_fisher_ratio"] = float(np.max((mu1 - mu0) ** 2 / (v0 + v1 + 1e-12)))
    out["centroid_distance"] = float(np.linalg.norm(mu1 - mu0))
    rng = np.random.RandomState(0)
    sub = rng.choice(len(X), size=min(4000, len(X)), replace=False)
    if len(np.unique(y[sub])) > 1:
        out["silhouette_score"] = float(silhouette_score(X[sub], y[sub]))

    if run is not None and ds is not None and ck is not None and i_fit is not None:
        sY, sX, T = ds["sY"], ds["sX"], ds["T"]
        comp = {}
        comp["rc500"] = _spec_and_fisher(X, y)
        comp["rc500"]["pretty"] = "Physical RC, raw state (500D)"

        Xdiff = ((ds["X_grid"] - ck["ground_mean"].reshape(1, 1, T)) / ck["std"].reshape(1, 1, T)
                 ).astype(np.float32)
        # Project each pixel onto top 60 single-pixel PCs (>99.96% variance) before forming 3x3 patch (540D)
        V60 = vecs_full[:, idx_full[:60]].astype(np.float32)
        Xdiff_60 = (Xdiff.reshape(-1, T) @ V60).reshape(sY, sX, 60)
        Pf_540 = D.build_patch_features(Xdiff_60, cfg["models"]["patch_size"]).reshape(-1, 9 * 60)[i_fit]
        # Exact total variance of full 4500D patch on i_fit
        pad_var = np.pad(Xdiff, ((1, 1), (1, 1), (0, 0)), mode="edge")
        tot_patch_var = 0.0
        for dy in range(3):
            for dx in range(3):
                sub_g = pad_var[dy:dy + sY, dx:dx + sX, :].reshape(-1, T)[i_fit]
                tot_patch_var += float(np.sum(sub_g.var(axis=0, ddof=1)))
        comp["patch4500"] = _spec_and_fisher(Pf_540, y, total_var_override=tot_patch_var)
        comp["patch4500"]["pretty"] = "Physical RC, 3x3 patch (4500D)"
        del Pf_540, Xdiff_60, Xdiff, pad_var

        for sz in [500, 50]:
            sp = os.path.join(run, "states", f"blocked_f1_esn{sz}.npy")
            if os.path.exists(sp):
                H = np.load(sp)[i_fit]
                comp[f"esn{sz}"] = _spec_and_fisher(H, y)
                comp[f"esn{sz}"]["pretty"] = f"Software ESN, {sz} neurons ({sz}D)"
                del H
        out["comparative_spaces"] = comp
    return out


def temporal_windows(Xn, y, i_fit, i_te, cfg, win=50, step=25, ds=None, ck=None, run=None):
    """Sliding temporal window: which part of the transient carries defect information."""
    from sklearn.linear_model import RidgeClassifier
    T = Xn.shape[1]
    centres, f1s, accs, aucs = [], [], [], []
    for s in range(0, T - win + 1, step):
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced", solver="cholesky")
        clf.fit(Xn[i_fit][:, s:s + win], y[i_fit])
        scores = clf.decision_function(Xn[i_te][:, s:s + win])
        yp = (scores >= 0.0).astype(int)
        centres.append(int(s + win // 2))
        f1s.append(float(f1_score(y[i_te], yp, average="macro", zero_division=0) * 100))
        accs.append(float(accuracy_score(y[i_te], yp) * 100))
        aucs.append(float(roc_auc_score(y[i_te], scores) * 100))

    res_tw = {
        "window_length": win, "step": step, "window_centre": centres,
        "macro_f1": f1s, "accuracy": accs, "roc_auc": aucs,
        "ground_mean_transient": Xn[i_fit][y[i_fit] == 0].mean(axis=0).tolist(),
        "ground_std_transient": Xn[i_fit][y[i_fit] == 0].std(axis=0).tolist(),
        "defect_mean_transient": Xn[i_fit][y[i_fit] == 1].mean(axis=0).tolist(),
    }

    if ds is not None and ck is not None:
        sY, sX = ds["sY"], ds["sX"]
        did_all = ds["defect_id_map"].ravel()
        diam_mm = cfg["dataset"]["defect_diameter_mm"]
        A0 = Xn[y == 0]
        mu0, v0, n0 = A0.mean(axis=0), A0.var(axis=0, ddof=1), len(A0)

        A1 = Xn[y == 1]
        mu1, v1, n1 = A1.mean(axis=0), A1.var(axis=0, ddof=1), len(A1)
        res_tw["fisher_per_sample_all"] = ((mu1 - mu0) ** 2 / (v0 + v1 + 1e-12)).tolist()
        wt_all = np.abs(mu1 - mu0) / np.sqrt(v0 / n0 + v1 / n1 + 1e-12)
        res_tw["welch_t_per_sample_all"] = wt_all.tolist()
        res_tw["max_welch_t_all"] = float(np.max(wt_all))
        res_tw["max_welch_t_sample_all"] = int(np.argmax(wt_all))

        for d_val, tag in [(10, "10mm"), (8, "8mm"), (6, "6mm")]:
            ids = [idx + 1 for idx, dm in enumerate(diam_mm) if dm == d_val]
            mask_d = np.isin(did_all, ids)
            if mask_d.sum() > 1:
                Ad = Xn[mask_d]
                mud, vd, nd = Ad.mean(axis=0), Ad.var(axis=0, ddof=1), len(Ad)
                res_tw[f"defect_{tag}_mean_transient"] = mud.tolist()
                res_tw[f"fisher_per_sample_{tag}"] = ((mud - mu0) ** 2 / (v0 + vd + 1e-12)).tolist()
                wt_d = np.abs(mud - mu0) / np.sqrt(v0 / n0 + vd / nd + 1e-12)
                res_tw[f"welch_t_per_sample_{tag}"] = wt_d.tolist()
                res_tw[f"max_welch_t_{tag}"] = float(np.max(wt_d))
                res_tw[f"max_welch_t_sample_{tag}"] = int(np.argmax(wt_d))

        # Sliding window for 3x3 patch and filtered patch on blocked protocol
        Xdiff = ((ds["X_grid"] - ck["ground_mean"].reshape(1, 1, T)) / ck["std"].reshape(1, 1, T)
                 ).astype(np.float32)
        pp = cfg["postprocess"]
        patch_f1, patch_auc, pfilt_f1, pfilt_auc = [], [], [], []
        for s in range(0, T - win + 1, step):
            Pw = D.build_patch_features(Xdiff[:, :, s:s + win], cfg["models"]["patch_size"]
                                        ).reshape(-1, 9 * win)
            clf = RidgeClassifier(alpha=10.0, class_weight="balanced", solver="cholesky")
            clf.fit(Pw[i_fit], y[i_fit])
            sc_all = clf.decision_function(Pw)
            sc_te = sc_all[i_te]
            yp = (sc_te >= 0.0).astype(int)
            patch_f1.append(float(f1_score(y[i_te], yp, average="macro", zero_division=0) * 100))
            patch_auc.append(float(roc_auc_score(y[i_te], sc_te) * 100))

            fmap = gaussian_filter(median_filter(sc_all.reshape(sY, sX), size=pp["median_size"]),
                                   sigma=pp["gaussian_sigma"]).ravel()
            sc_fte = fmap[i_te]
            yp_f = (sc_fte >= 0.0).astype(int)
            pfilt_f1.append(float(f1_score(y[i_te], yp_f, average="macro", zero_division=0) * 100))
            pfilt_auc.append(float(roc_auc_score(y[i_te], sc_fte) * 100))
            del Pw
        res_tw["patch4500_macro_f1"] = patch_f1
        res_tw["patch4500_roc_auc"] = patch_auc
        res_tw["patch4500_filt_macro_f1"] = pfilt_f1
        res_tw["patch4500_filt_roc_auc"] = pfilt_auc
        del Xdiff

        # Also compute sliding window on random_optimistic and loco_guard (pooled/mean)
        if run is not None:
            by_proto = {"blocked": {
                "rc500": {"macro_f1": f1s, "roc_auc": aucs},
                "patch4500": {"macro_f1": patch_f1, "roc_auc": patch_auc},
                "patch4500_filt": {"macro_f1": pfilt_f1, "roc_auc": pfilt_auc},
            }}
            # Random split (use stratified 3500-pixel fit subsample for 450D patch speed)
            mk_r = np.load(os.path.join(run, "masks", "random_optimistic_f1.npz"))
            ck_r = joblib.load(os.path.join(run, "checkpoints", "random_optimistic_f1.joblib"))
            ifit_r = np.flatnonzero(mk_r["fit_mask"].ravel())
            ite_r = np.flatnonzero(mk_r["test_mask"].ravel())
            rng_tw = np.random.RandomState(42)
            pos_r = ifit_r[y[ifit_r] == 1]
            neg_r = ifit_r[y[ifit_r] == 0]
            ifit_r_sub = np.concatenate([pos_r, rng_tw.choice(neg_r, size=min(3200, len(neg_r)), replace=False)])
            Xn_r = (ds["X_grid"].reshape(-1, T) - ck_r["mean"]) / ck_r["std"]
            Xdiff_r = ((ds["X_grid"] - ck_r["ground_mean"].reshape(1, 1, T)) / ck_r["std"].reshape(1, 1, T)
                       ).astype(np.float32)
            rf1_rc, rauc_rc, rf1_pf, rauc_pf = [], [], [], []
            for s in range(0, T - win + 1, step):
                clf = RidgeClassifier(alpha=1.0, class_weight="balanced", solver="cholesky")
                clf.fit(Xn_r[ifit_r][:, s:s + win], y[ifit_r])
                sc = clf.decision_function(Xn_r[ite_r][:, s:s + win])
                rf1_rc.append(float(f1_score(y[ite_r], (sc >= 0.0).astype(int), average="macro", zero_division=0) * 100))
                rauc_rc.append(float(roc_auc_score(y[ite_r], sc) * 100))

                Pw = D.build_patch_features(Xdiff_r[:, :, s:s + win], cfg["models"]["patch_size"]
                                            ).reshape(-1, 9 * win)
                clf_p = RidgeClassifier(alpha=10.0, class_weight="balanced", solver="cholesky")
                clf_p.fit(Pw[ifit_r_sub], y[ifit_r_sub])
                sc_all = clf_p.decision_function(Pw)
                fmap = gaussian_filter(median_filter(sc_all.reshape(sY, sX), size=pp["median_size"]),
                                       sigma=pp["gaussian_sigma"]).ravel()
                sc_fte = fmap[ite_r]
                rf1_pf.append(float(f1_score(y[ite_r], (sc_fte >= 0.0).astype(int), average="macro", zero_division=0) * 100))
                rauc_pf.append(float(roc_auc_score(y[ite_r], sc_fte) * 100))
                del Pw
            by_proto["random_optimistic"] = {
                "rc500": {"macro_f1": rf1_rc, "roc_auc": rauc_rc},
                "patch4500_filt": {"macro_f1": rf1_pf, "roc_auc": rauc_pf},
            }
            del Xn_r, Xdiff_r

            # LOCO (mean over 9 folds, building patch features once per window s)
            loco_folds_info = []
            for fn_idx in range(1, 10):
                mk_l = np.load(os.path.join(run, "masks", f"loco_guard_f{fn_idx}.npz"))
                ck_l = joblib.load(os.path.join(run, "checkpoints", f"loco_guard_f{fn_idx}.joblib"))
                ifit_l = np.flatnonzero(mk_l["fit_mask"].ravel())
                ite_l = np.flatnonzero(mk_l["test_mask"].ravel())
                pos_l = ifit_l[y[ifit_l] == 1]
                neg_l = ifit_l[y[ifit_l] == 0]
                ifit_l_sub = np.concatenate([pos_l, rng_tw.choice(neg_l, size=min(2500, len(neg_l)), replace=False)])
                # Bounding box around test_mask (+6px margin for patch + spatial filter)
                ys, xs = np.nonzero(mk_l["test_mask"])
                y0, y1 = max(0, int(ys.min()) - 6), min(sY, int(ys.max()) + 7)
                x0, x1 = max(0, int(xs.min()) - 6), min(sX, int(xs.max()) + 7)
                sub_te_mask = mk_l["test_mask"][y0:y1, x0:x1].ravel()
                loco_folds_info.append((ck_l, ifit_l, ifit_l_sub, ite_l, (y0, y1, x0, x1, sub_te_mask)))

            # Use fold 1 normalisation for shared patch window construction
            ck1 = loco_folds_info[0][0]
            Xdiff_l0 = ((ds["X_grid"] - ck1["ground_mean"].reshape(1, 1, T)) / ck1["std"].reshape(1, 1, T)
                        ).astype(np.float32)
            Xn_l0 = (ds["X_grid"].reshape(-1, T) - ck1["mean"]) / ck1["std"]

            lf1_rc_folds = [[] for _ in range(9)]
            lauc_rc_folds = [[] for _ in range(9)]
            lf1_pf_folds = [[] for _ in range(9)]
            lauc_pf_folds = [[] for _ in range(9)]

            for s in range(0, T - win + 1, step):
                Xw = Xn_l0[:, s:s + win]
                Pw_grid = D.build_patch_features(Xdiff_l0[:, :, s:s + win], cfg["models"]["patch_size"])
                Pw_flat = Pw_grid.reshape(-1, 9 * win)
                for fi, (ck_l, ifit_l, ifit_l_sub, ite_l, (y0, y1, x0, x1, sub_te_mask)) in enumerate(loco_folds_info):
                    clf = RidgeClassifier(alpha=1.0, class_weight="balanced", solver="cholesky")
                    clf.fit(Xw[ifit_l], y[ifit_l])
                    sc = clf.decision_function(Xw[ite_l])
                    lf1_rc_folds[fi].append(float(f1_score(y[ite_l], (sc >= 0.0).astype(int), average="macro", zero_division=0) * 100))
                    lauc_rc_folds[fi].append(float(roc_auc_score(y[ite_l], sc) * 100))

                    clf_p = RidgeClassifier(alpha=10.0, class_weight="balanced", solver="cholesky")
                    clf_p.fit(Pw_flat[ifit_l_sub], y[ifit_l_sub])
                    sub_pw = Pw_grid[y0:y1, x0:x1, :].reshape(-1, 9 * win)
                    sc_box = clf_p.decision_function(sub_pw).reshape(y1 - y0, x1 - x0)
                    fmap_box = gaussian_filter(median_filter(sc_box, size=pp["median_size"]),
                                               sigma=pp["gaussian_sigma"]).ravel()
                    sc_fte = fmap_box[sub_te_mask]
                    lf1_pf_folds[fi].append(float(f1_score(y[ite_l], (sc_fte >= 0.0).astype(int), average="macro", zero_division=0) * 100))
                    lauc_pf_folds[fi].append(float(roc_auc_score(y[ite_l], sc_fte) * 100))
                del Pw_grid, Pw_flat
            del Xdiff_l0, Xn_l0
            by_proto["loco_guard"] = {
                "rc500": {
                    "macro_f1": np.mean(lf1_rc_folds, axis=0).tolist(),
                    "roc_auc": np.mean(lauc_rc_folds, axis=0).tolist(),
                },
                "patch4500_filt": {
                    "macro_f1": np.mean(lf1_pf_folds, axis=0).tolist(),
                    "roc_auc": np.mean(lauc_pf_folds, axis=0).tolist(),
                },
            }
            res_tw["by_protocol"] = by_proto

    return res_tw


def _fast_esn_states(esn, X):
    """Identical mathematical output to SoftwareESN.states(X), computed in float32."""
    n, T = X.shape
    out = np.empty((n, esn.size), dtype=np.float32)
    Wr = esn.W_res.T.astype(np.float32)
    alpha = np.float32(esn.leak_rate)
    one_m_alpha = np.float32(1.0 - esn.leak_rate)
    block = 4096
    w_in = esn.W_in.astype(np.float32)
    X32 = np.asarray(X, dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        Xb = X32[s:e]
        x = np.zeros((e - s, esn.size), dtype=np.float32)
        acc = np.zeros((e - s, esn.size), dtype=np.float32)
        for t in range(T):
            pre = Xb[:, t : t + 1] * w_in + x @ Wr
            x = one_m_alpha * x + alpha * np.tanh(pre)
            acc += x
        out[s:e] = acc / np.float32(T)
    return out


def _fast_patch_decision(clf, Xd_grid, patch_size=3):
    """Exact linear decision function of RidgeClassifier on 3x3 patch features without 460MB allocation."""
    p = patch_size // 2
    sY, sX, T = Xd_grid.shape
    n_shifts = patch_size * patch_size
    W = clf.coef_.ravel().reshape(n_shifts, T).T  # (T, 9)
    b = float(np.asarray(clf.intercept_).ravel()[0])
    proj = (Xd_grid.reshape(-1, T) @ W).reshape(sY, sX, n_shifts)
    pad = np.pad(proj, ((p, p), (p, p), (0, 0)), mode="edge")
    acc = np.full((sY, sX), b, dtype=np.float64)
    c = 0
    for dy in range(patch_size):
        for dx in range(patch_size):
            acc += pad[dy : dy + sY, dx : dx + sX, c]
            c += 1
    return acc.ravel()


def update_extra_metrics(run, ds, cfg):
    """Compute expanded reservoir_characterisation.json and multi-protocol noise_robustness.json."""
    sY, sX, T = ds["sY"], ds["sX"], ds["T"]
    y_all = ds["label_map"].ravel()

    # 1. Expanded reservoir_characterisation.json on blocked_f1
    log("computing expanded reservoir_characterisation.json ...")
    ck_b = joblib.load(os.path.join(run, "checkpoints", "blocked_f1.joblib"))
    mk_b = np.load(os.path.join(run, "masks", "blocked_f1.npz"))
    i_fit_b = np.flatnonzero(mk_b["fit_mask"].ravel())
    i_te_b = np.flatnonzero(mk_b["test_mask"].ravel())
    Xn0_b = (ds["X_grid"].reshape(-1, T) - ck_b["mean"]) / ck_b["std"]

    res = characterise(Xn0_b[i_fit_b], y_all[i_fit_b], cfg,
                       run=run, ds=ds, ck=ck_b, i_fit=i_fit_b)
    res["temporal_windows"] = temporal_windows(Xn0_b, y_all, i_fit_b, i_te_b, cfg,
                                               ds=ds, ck=ck_b, run=run)
    with open(os.path.join(run, "metrics", "reservoir_characterisation.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("  reservoir_characterisation.json written")

    # 2. Expanded multi-protocol, all-model noise_robustness.json
    log("computing multi-protocol noise_robustness.json ...")
    ne = cfg["noise_evaluation"]
    snrs = ne["snr_db"]
    n_rep = ne["n_repeats"]
    all_models = MODEL_ORDER

    # First preserve exact original blocked noise["models"] for backward compatibility
    nr_path = os.path.join(run, "metrics", "noise_robustness.json")
    noise_out = {"snr_db": snrs, "n_repeats": n_rep, "models": {}, "by_protocol": {}}
    if os.path.exists(nr_path):
        with open(nr_path) as f:
            prev_nr = json.load(f)
        if "models" in prev_nr and prev_nr["models"]:
            noise_out["models"] = prev_nr["models"]
    if not noise_out["models"]:
        rng_orig = np.random.RandomState(cfg["protocols"]["random_seed"])
        sig_pow_b = float(np.mean(Xn0_b ** 2))
        orig_targets = ne.get("models", ["rc500", "esn50", "patch4500_filt"])
        for n in orig_targets:
            noise_out["models"][n] = {"macro_f1_mean": [], "macro_f1_std": [],
                                      "accuracy_mean": [], "pretty": PRETTY[n]}
        for snr in snrs:
            sd = np.sqrt(sig_pow_b / (10.0 ** (snr / 10.0)))
            acc_r = {n: [] for n in orig_targets}
            f1_r = {n: [] for n in orig_targets}
            for _ in range(n_rep):
                Xnz = Xn0_b + rng_orig.normal(0, sd, size=Xn0_b.shape)
                for n in orig_targets:
                    if n == "rc500":
                        s = ck_b["models"]["rc500"]["clf"].decision_function(Xnz[i_te_b])
                    elif n.startswith("esn"):
                        s = ck_b["models"][n]["clf"].decision_function(
                            _fast_esn_states(ck_b["models"][n]["esn"], Xnz[i_te_b]))
                    else:
                        Xd = ((Xnz * ck_b["std"] + ck_b["mean"] - ck_b["ground_mean"])
                              / ck_b["std"]).reshape(sY, sX, T).astype(np.float32)
                        sm = _fast_patch_decision(
                            ck_b["models"]["patch4500"]["clf"], Xd, cfg["models"]["patch_size"]
                        ).reshape(sY, sX)
                        pp = cfg["postprocess"]
                        s = gaussian_filter(median_filter(sm, size=pp["median_size"]),
                                            sigma=pp["gaussian_sigma"]).ravel()[i_te_b]
                        del Xd
                    th = ck_b["models"][n]["threshold_val"]
                    yp = (s >= th).astype(int)
                    f1_r[n].append(f1_score(y_all[i_te_b], yp, average="macro", zero_division=0) * 100)
                    acc_r[n].append(accuracy_score(y_all[i_te_b], yp) * 100)
                del Xnz
            for n in orig_targets:
                noise_out["models"][n]["macro_f1_mean"].append(float(np.mean(f1_r[n])))
                noise_out["models"][n]["macro_f1_std"].append(float(np.std(f1_r[n])))
                noise_out["models"][n]["accuracy_mean"].append(float(np.mean(acc_r[n])))

    # Now compute comprehensive 7-model noise sweep across all 3 protocols
    proto_tags = {
        "loco_guard": [f"loco_guard_f{i}" for i in range(1, 10)],
        "blocked": ["blocked_f1"],
        "random_optimistic": ["random_optimistic_f1"],
    }
    X_flat = ds["X_grid"].reshape(-1, T)
    pp = cfg["postprocess"]
    rng_sub = np.random.RandomState(42)

    # Global stratified subset for ESN50/ESN500 so recurrent loop runs ONCE per repeat instead of 11x
    all_pos = np.flatnonzero(y_all == 1)
    all_neg = np.flatnonzero(y_all == 0)
    neg_pool = rng_sub.choice(all_neg, size=450, replace=False)
    global_esn_idx = np.sort(np.concatenate([all_pos, neg_pool]))
    global_pos_map = np.full(len(y_all), -1, dtype=np.int32)
    global_pos_map[global_esn_idx] = np.arange(len(global_esn_idx), dtype=np.int32)

    loaded = {}
    for proto, tags in proto_tags.items():
        loaded[proto] = []
        for t_ in tags:
            ck_ = joblib.load(os.path.join(run, "checkpoints", f"{t_}.joblib"))
            mk_ = np.load(os.path.join(run, "masks", f"{t_}.npz"))
            i_te_ = np.flatnonzero(mk_["test_mask"].ravel())
            y_te_ = y_all[i_te_]
            Xn0_te = ((X_flat[i_te_] - ck_["mean"]) / ck_["std"]).astype(np.float32)

            # Bounding box around test_mask (+6px margin for 3x3 patch + median/gaussian filter)
            ys, xs = np.nonzero(mk_["test_mask"])
            y0, y1 = max(0, int(ys.min()) - 6), min(sY, int(ys.max()) + 7)
            x0, x1 = max(0, int(xs.min()) - 6), min(sX, int(xs.max()) + 7)
            sub_te_mask = mk_["test_mask"][y0:y1, x0:x1].ravel()
            Xd0_box = ((ds["X_grid"][y0:y1, x0:x1, :] - ck_["ground_mean"].reshape(1, 1, T))
                       / ck_["std"].reshape(1, 1, T)).astype(np.float32)
            sc_patch0_box = _fast_patch_decision(
                ck_["models"]["patch4500"]["clf"], Xd0_box, cfg["models"]["patch_size"]
            )

            # Map test indices that are in global_esn_idx
            in_g = global_pos_map[i_te_] >= 0
            g_slots = global_pos_map[i_te_[in_g]]
            y_g = y_te_[in_g]
            n_pos_te = int(np.sum(y_te_ == 1))
            n_neg_te = int(np.sum(y_te_ == 0))
            n_neg_g = max(1, int(np.sum(y_g == 0)))
            esn_w = np.where(y_g == 1, 1.0, float(n_neg_te) / float(n_neg_g)).astype(np.float64)

            loaded[proto].append({
                "tag": t_, "ck": ck_, "i_te": i_te_, "y_te": y_te_, "Xn0_te": Xn0_te,
                "box": (y0, y1, x0, x1, sub_te_mask), "sc_patch0_box": sc_patch0_box,
                "g_slots": g_slots, "y_g": y_g, "esn_w": esn_w,
            })

    for proto in ["loco_guard", "blocked", "random_optimistic"]:
        noise_out["by_protocol"][proto] = {
            n: {"macro_f1_mean": [], "macro_f1_std": [],
                "roc_auc_mean": [], "roc_auc_std": [],
                "accuracy_mean": [], "pretty": PRETTY[n]}
            for n in all_models
        }

    Xn0_global_esn = Xn0_b[global_esn_idx].astype(np.float32)
    esn50_ref = ck_b["models"]["esn50"]["esn"]
    esn500_ref = ck_b["models"]["esn500"]["esn"]

    rng_all = np.random.RandomState(cfg["protocols"]["random_seed"])
    for snr in snrs:
        t0_s = time.time()
        sd_unit = np.float32(np.sqrt(1.0 / (10.0 ** (snr / 10.0))))
        rep_metrics = {proto: {n: {"f1": [], "auc": [], "acc": []} for n in all_models}
                       for proto in proto_tags}
        for r_idx in range(n_rep):
            Z_grid = rng_all.normal(0.0, sd_unit, size=(sY, sX, T)).astype(np.float32)
            Z_flat = Z_grid.reshape(-1, T)
            Xn_esn_noisy = Xn0_global_esn + Z_flat[global_esn_idx]
            H50_global = _fast_esn_states(esn50_ref, Xn_esn_noisy)
            H500_global = _fast_esn_states(esn500_ref, Xn_esn_noisy)

            for proto, fold_list in loaded.items():
                fold_f1 = {n: [] for n in all_models}
                fold_auc = {n: [] for n in all_models}
                fold_acc = {n: [] for n in all_models}
                for finfo in fold_list:
                    ck_ = finfo["ck"]
                    i_te_ = finfo["i_te"]
                    y_te = finfo["y_te"]
                    Xn_te = finfo["Xn0_te"] + Z_flat[i_te_]

                    for n in ["peak1", "pca10", "rc500"]:
                        m_obj = ck_["models"][n]
                        sc = m_obj["clf"].decision_function(m_obj["feature"].transform(Xn_te))
                        th = m_obj["threshold_val"]
                        yp = (sc >= th).astype(int)
                        fold_f1[n].append(f1_score(y_te, yp, average="macro", zero_division=0) * 100)
                        fold_auc[n].append(roc_auc_score(y_te, sc) * 100)
                        fold_acc[n].append(accuracy_score(y_te, yp) * 100)

                    # ESN50 & ESN500 sliced from single global ESN pass
                    g_slots = finfo["g_slots"]
                    y_g = finfo["y_g"]
                    sub_w = finfo["esn_w"]
                    for n_esn, H_g in [("esn50", H50_global), ("esn500", H500_global)]:
                        m_e = ck_["models"][n_esn]
                        sc_e = m_e["clf"].decision_function(H_g[g_slots])
                        th_e = m_e["threshold_val"]
                        yp_e = (sc_e >= th_e).astype(int)
                        fold_f1[n_esn].append(f1_score(y_g, yp_e, sample_weight=sub_w, average="macro", zero_division=0) * 100)
                        fold_auc[n_esn].append(roc_auc_score(y_g, sc_e, sample_weight=sub_w) * 100)
                        fold_acc[n_esn].append(accuracy_score(y_g, yp_e, sample_weight=sub_w) * 100)

                    # Patch4500 & Patch4500_filt on bounding box around test_mask
                    y0, y1, x0, x1, sub_te_mask = finfo["box"]
                    clf_p = ck_["models"]["patch4500"]["clf"]
                    b_p = float(np.asarray(clf_p.intercept_).ravel()[0])
                    sc_noise_box = _fast_patch_decision(clf_p, Z_grid[y0:y1, x0:x1, :], cfg["models"]["patch_size"]) - b_p
                    sc_patch_box = finfo["sc_patch0_box"] + sc_noise_box
                    sc_p = sc_patch_box[sub_te_mask]
                    th_p = ck_["models"]["patch4500"]["threshold_val"]
                    yp_p = (sc_p >= th_p).astype(int)
                    fold_f1["patch4500"].append(f1_score(y_te, yp_p, average="macro", zero_division=0) * 100)
                    fold_auc["patch4500"].append(roc_auc_score(y_te, sc_p) * 100)
                    fold_acc["patch4500"].append(accuracy_score(y_te, yp_p) * 100)

                    fmap_box = gaussian_filter(median_filter(sc_patch_box.reshape(y1 - y0, x1 - x0), size=pp["median_size"]),
                                               sigma=pp["gaussian_sigma"]).ravel()
                    sc_pf = fmap_box[sub_te_mask]
                    th_pf = ck_["models"]["patch4500_filt"]["threshold_val"]
                    yp_pf = (sc_pf >= th_pf).astype(int)
                    fold_f1["patch4500_filt"].append(f1_score(y_te, yp_pf, average="macro", zero_division=0) * 100)
                    fold_auc["patch4500_filt"].append(roc_auc_score(y_te, sc_pf) * 100)
                    fold_acc["patch4500_filt"].append(accuracy_score(y_te, yp_pf) * 100)
                for n in all_models:
                    rep_metrics[proto][n]["f1"].append(float(np.mean(fold_f1[n])))
                    rep_metrics[proto][n]["auc"].append(float(np.mean(fold_auc[n])))
                    rep_metrics[proto][n]["acc"].append(float(np.mean(fold_acc[n])))
            del Z_grid, Z_flat
        for proto in proto_tags:
            for n in all_models:
                noise_out["by_protocol"][proto][n]["macro_f1_mean"].append(float(np.mean(rep_metrics[proto][n]["f1"])))
                noise_out["by_protocol"][proto][n]["macro_f1_std"].append(float(np.std(rep_metrics[proto][n]["f1"])))
                noise_out["by_protocol"][proto][n]["roc_auc_mean"].append(float(np.mean(rep_metrics[proto][n]["auc"])))
                noise_out["by_protocol"][proto][n]["roc_auc_std"].append(float(np.std(rep_metrics[proto][n]["auc"])))
                noise_out["by_protocol"][proto][n]["accuracy_mean"].append(float(np.mean(rep_metrics[proto][n]["acc"])))
        log(f"  SNR {snr:>5} dB across all 3 protocols & 7 models done ({time.time()-t0_s:.1f}s)")

    with open(os.path.join(run, "metrics", "noise_robustness.json"), "w") as f:
        json.dump(noise_out, f, indent=2)
    log("  noise_robustness.json written")


def fix_esn_noise_and_welch(run, ds, cfg):
    """Fast update of Welch peak metadata and per-fold guaranteed-stratified ESN50/ESN500 noise metrics."""
    sY, sX, T = ds["sY"], ds["sX"], ds["T"]
    y_all = ds["label_map"].ravel()

    rc_path = os.path.join(run, "metrics", "reservoir_characterisation.json")
    with open(rc_path) as f:
        res = json.load(f)
    tw = res["temporal_windows"]
    for tag in ["all", "10mm", "8mm", "6mm"]:
        k_w = f"welch_t_per_sample_{tag}"
        if k_w in tw:
            arr_w = np.asarray(tw[k_w])
            tw[f"max_welch_t_{tag}"] = float(np.max(arr_w))
            tw[f"max_welch_t_sample_{tag}"] = int(np.argmax(arr_w))
    with open(rc_path, "w") as f:
        json.dump(res, f, indent=2)
    log("  updated max_welch_t metadata in reservoir_characterisation.json")

    nr_path = os.path.join(run, "metrics", "noise_robustness.json")
    with open(nr_path) as f:
        noise_out = json.load(f)

    ne = cfg["noise_evaluation"]
    snrs = ne["snr_db"]
    n_rep = ne["n_repeats"]
    proto_tags = {
        "loco_guard": [f"loco_guard_f{i}" for i in range(1, 10)],
        "blocked": ["blocked_f1"],
        "random_optimistic": ["random_optimistic_f1"],
    }
    X_flat = ds["X_grid"].reshape(-1, T)
    ck_b = joblib.load(os.path.join(run, "checkpoints", "blocked_f1.joblib"))
    Xn0_b = ((X_flat - ck_b["mean"]) / ck_b["std"]).astype(np.float32)

    rng_sub = np.random.RandomState(12345)
    fold_subsets = {}
    pool_indices = set()
    for proto, tags in proto_tags.items():
        fold_subsets[proto] = []
        for t_ in tags:
            ck_ = joblib.load(os.path.join(run, "checkpoints", f"{t_}.joblib"))
            mk_ = np.load(os.path.join(run, "masks", f"{t_}.npz"))
            i_te_ = np.flatnonzero(mk_["test_mask"].ravel())
            y_te_ = y_all[i_te_]
            pos_te = i_te_[y_te_ == 1]
            neg_te = i_te_[y_te_ == 0]
            pos_keep = rng_sub.choice(pos_te, size=min(20, len(pos_te)), replace=False)
            neg_keep = rng_sub.choice(neg_te, size=min(25, len(neg_te)), replace=False)
            sub_global_idx = np.concatenate([pos_keep, neg_keep])
            pool_indices.update(sub_global_idx.tolist())
            fold_subsets[proto].append({
                "ck": ck_,
                "sub_global_idx": sub_global_idx,
                "n_pos_te": int(len(pos_te)),
                "n_neg_te": int(len(neg_te)),
            })

    global_esn_idx = np.array(sorted(pool_indices), dtype=np.int32)
    global_pos_map = np.full(len(y_all), -1, dtype=np.int32)
    global_pos_map[global_esn_idx] = np.arange(len(global_esn_idx), dtype=np.int32)

    for proto in proto_tags:
        for finfo in fold_subsets[proto]:
            sub_g = finfo["sub_global_idx"]
            finfo["g_slots"] = global_pos_map[sub_g]
            y_g = y_all[sub_g]
            finfo["y_g"] = y_g
            w_pos = float(finfo["n_pos_te"]) / max(1, int(np.sum(y_g == 1)))
            w_neg = float(finfo["n_neg_te"]) / max(1, int(np.sum(y_g == 0)))
            finfo["esn_w"] = np.where(y_g == 1, w_pos, w_neg).astype(np.float64)

    Xn0_global_esn = Xn0_b[global_esn_idx]
    esn50_ref = ck_b["models"]["esn50"]["esn"]
    esn500_ref = ck_b["models"]["esn500"]["esn"]

    for proto in proto_tags:
        for n_esn in ["esn50", "esn500"]:
            noise_out["by_protocol"][proto][n_esn] = {
                "macro_f1_mean": [], "macro_f1_std": [],
                "roc_auc_mean": [], "roc_auc_std": [],
                "accuracy_mean": [], "pretty": PRETTY[n_esn],
            }

    rng_all = np.random.RandomState(cfg["protocols"]["random_seed"])
    for snr in snrs:
        t0_s = time.time()
        sd_unit = np.float32(np.sqrt(1.0 / (10.0 ** (snr / 10.0))))
        rep_m = {proto: {n: {"f1": [], "auc": [], "acc": []} for n in ["esn50", "esn500"]}
                 for proto in proto_tags}
        for _ in range(n_rep):
            Z_sub = rng_all.normal(0.0, sd_unit, size=(len(global_esn_idx), T)).astype(np.float32)
            Xn_noisy = Xn0_global_esn + Z_sub
            H50_g = _fast_esn_states(esn50_ref, Xn_noisy)
            H500_g = _fast_esn_states(esn500_ref, Xn_noisy)
            for proto, flist in fold_subsets.items():
                f1_f = {"esn50": [], "esn500": []}
                auc_f = {"esn50": [], "esn500": []}
                acc_f = {"esn50": [], "esn500": []}
                for finfo in flist:
                    ck_ = finfo["ck"]
                    g_slots = finfo["g_slots"]
                    y_g = finfo["y_g"]
                    sub_w = finfo["esn_w"]
                    for n_esn, H_g in [("esn50", H50_g), ("esn500", H500_g)]:
                        m_e = ck_["models"][n_esn]
                        sc_e = m_e["clf"].decision_function(H_g[g_slots])
                        th_e = m_e["threshold_val"]
                        yp_e = (sc_e >= th_e).astype(int)
                        f1_f[n_esn].append(f1_score(y_g, yp_e, sample_weight=sub_w, average="macro", zero_division=0) * 100)
                        auc_f[n_esn].append(roc_auc_score(y_g, sc_e, sample_weight=sub_w) * 100)
                        acc_f[n_esn].append(accuracy_score(y_g, yp_e, sample_weight=sub_w) * 100)
                for n_esn in ["esn50", "esn500"]:
                    rep_m[proto][n_esn]["f1"].append(float(np.mean(f1_f[n_esn])))
                    rep_m[proto][n_esn]["auc"].append(float(np.mean(auc_f[n_esn])))
                    rep_m[proto][n_esn]["acc"].append(float(np.mean(acc_f[n_esn])))
        for proto in proto_tags:
            for n_esn in ["esn50", "esn500"]:
                noise_out["by_protocol"][proto][n_esn]["macro_f1_mean"].append(float(np.mean(rep_m[proto][n_esn]["f1"])))
                noise_out["by_protocol"][proto][n_esn]["macro_f1_std"].append(float(np.std(rep_m[proto][n_esn]["f1"])))
                noise_out["by_protocol"][proto][n_esn]["roc_auc_mean"].append(float(np.mean(rep_m[proto][n_esn]["auc"])))
                noise_out["by_protocol"][proto][n_esn]["roc_auc_std"].append(float(np.std(rep_m[proto][n_esn]["auc"])))
                noise_out["by_protocol"][proto][n_esn]["accuracy_mean"].append(float(np.mean(rep_m[proto][n_esn]["acc"])))
        log(f"  ESN SNR {snr:>5} dB done ({time.time()-t0_s:.1f}s)")

    with open(nr_path, "w") as f:
        json.dump(noise_out, f, indent=2)
    log("  updated noise_robustness.json written")


if __name__ == "__main__":
    main()
