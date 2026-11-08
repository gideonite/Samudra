"""FINAL test-split table. The study's only use of the test split: no selection may be based on it.
Usage: python final_test.py   (single process, 1 thread). Writes runs/_final_test.json and final_table.md.
For each saved run: load model.pt + args.json, re-evaluate on val and require it to match the logged `final` val metric
(1e-6), then evaluate on test. Runs are grouped by args.json (dataset, arm, n_train, seed, non-default flags), not by tag."""
import glob, json, math, os, sys
from pathlib import Path
import numpy as np, torch
import prepare, train

HERE = Path(__file__).parent
TOL = 1e-6
DFLT = vars(train.get_args([]))
IGN = {"tag", "out_dir", "dataset", "arm", "n_train", "seed", "log_grad_norms"}  # bookkeeping-only flags
CLOBBERED_SUFFIX = "c084f4f"  # Phase-2 screen-1 dirs whose tags had no flag hash: overwritten, exclude
WARM = {"warm_frac": 0.5, "ensemble_topk": 3}
ARMS = ["baseline", "stochdepth", "mask_only", "reweight_only", "pcgb", "pcgb_warm"]
COLS = [("ks1d", 128), ("ks1d", 512), ("ks1d", 2048), ("kolmo2d", 128), ("kolmo2d", 512), ("kolmo2d", 2048),
        ("denoise", 512), ("grayscott2d", 512)]


def classify(a):
    """-> row label or None if the config is not part of the final table."""
    nd = {k: a[k] for k in a if k in DFLT and k not in IGN and a[k] != DFLT[k]}
    if not nd:
        return a["arm"]
    if a["arm"] == "pcgb" and nd == WARM:
        return "pcgb_warm"
    return None


def wanted(ds, arm, n):
    if (ds, n) not in COLS: return False
    if ds in ("denoise", "grayscott2d"):
        return arm in ("baseline", "stochdepth", "reweight_only", "pcgb", "pcgb_warm")
    if n == 128:
        return arm in ("baseline", "stochdepth", "pcgb", "pcgb_warm")
    return arm in ARMS if n != 2048 or True else False


def same(x, y, tol):
    return (math.isinf(x) and math.isinf(y) and x == y) or abs(x - y) <= tol


def main():
    torch.set_num_threads(1)
    skipped, found = [], {}
    for d in sorted(glob.glob(str(HERE / "runs" / "*"))):
        b = os.path.basename(d)
        if not (Path(d) / "args.json").exists(): continue
        a = json.loads((Path(d) / "args.json").read_text())
        label = classify(a)
        if label is None or not wanted(a["dataset"], label, a["n_train"]): continue
        if b.split("_")[-1] == CLOBBERED_SUFFIX and not b.startswith("calib_"):
            skipped.append((b, "clobbered Phase-2 screen-1 dir (no flag-hash suffix)")); continue
        fin = [json.loads(l) for l in open(Path(d) / "log.jsonl") if '"type": "final"' in l] if (Path(d) / "log.jsonl").exists() else []
        if not (Path(d) / "model.pt").exists() or not fin:
            skipped.append((b, "no model.pt / no final row (unfinished)")); continue
        key = (a["dataset"], a["n_train"], label, a["seed"])
        found.setdefault(key, []).append((b, a, fin[-1]))
    for k, v in found.items():
        assert len(v) == 1, ("duplicate runs for config", k, [x[0] for x in v])

    runs = []
    for ds in sorted({k[0] for k in found}):
        data = prepare.load(ds); meta = data["meta"]
        for key in sorted(k for k in found if k[0] == ds):
            b, a, fin = found[key][0]
            m = train.build_model(meta, a["width"], a["drop_mode"])
            m.load_state_dict(torch.load(HERE / "runs" / b / "model.pt", map_location="cpu")); m.eval()
            with torch.no_grad():
                ev = prepare.evaluate(m, ds, "val")
            if not same(ev["metric"], fin["metric"], TOL):
                skipped.append((b, f"val mismatch: re-eval {ev['metric']!r} vs logged {fin['metric']!r}")); continue
            with torch.no_grad():
                te = prepare.evaluate(m, ds, "test")
            runs.append(dict(run=b, dataset=ds, n_train=key[1], arm=key[2], seed=key[3], val_metric=ev["metric"],
                             val_onestep_mse=ev["onestep_mse"], test_metric=te["metric"], test_onestep_mse=te["onestep_mse"],
                             test_finite=bool(te["finite"]), test_per_step_nrmse=te["per_step_nrmse"],
                             val_logged_onestep_mse=fin.get("onestep_mse")))
            print(b, f"val {ev['metric']:.5f} test {te['metric']:.5f}", flush=True)
    (HERE / "runs" / "_final_test.json").write_text(json.dumps(dict(tol=TOL, runs=runs, skipped=skipped), indent=1))

    # table + rank agreement
    def cell_stats(ds, n, arm):
        r = [x for x in runs if (x["dataset"], x["n_train"], x["arm"]) == (ds, n, arm)]
        return r
    lines = ["| arm | " + " | ".join(f"{ds} n={n}" for ds, n in COLS) + " |", "|---|" + "---|" * len(COLS)]
    means = {c: {} for c in COLS}; nonfinite = []
    for arm in ARMS:
        cells = []
        for ds, n in COLS:
            r = cell_stats(ds, n, arm)
            if not r: cells.append("–"); continue
            t = np.array([x["test_metric"] for x in r]); v = np.array([x["val_metric"] for x in r])
            for x in r:
                if not x["test_finite"]: nonfinite.append(x["run"])
            sd = lambda z: z.std(ddof=1) if len(z) > 1 else float("nan")
            s = f"{t.mean():.3f} ± {sd(t):.3f} ({v.mean():.3f})"
            if len(r) != 3: s += f" [k={len(r)}]"
            cells.append(s); means[(ds, n)][arm] = (v.mean(), t.mean())
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    from scipy.stats import spearmanr, kendalltau
    rk = ["", "Val→test rank agreement per column (over arms, using seed-mean metrics; lower = better):", ""]
    for c in COLS:
        mm = means[c]; arms = list(mm)
        if len(arms) < 3: continue
        v = [mm[x][0] for x in arms]; t = [mm[x][1] for x in arms]
        if not all(np.isfinite(v + t)): rk.append(f"- {c[0]} n={c[1]}: non-finite means, skipped"); continue
        vo = [x for _, x in sorted(zip(v, arms))]; to = [x for _, x in sorted(zip(t, arms))]
        rk.append(f"- {c[0]} n={c[1]}: Spearman {spearmanr(v, t)[0]:.2f}, Kendall {kendalltau(v, t)[0]:.2f}, same best arm: "
                  f"{vo[0] == to[0]} (val {vo[0]}, test {to[0]}); val order {' < '.join(vo)}; test order {' < '.join(to)}")
    notes = ["", "Cells: test mean ± sample std over seeds (val mean). k = number of seeds when not 3. Metric = 10-step rollout nRMSE "
             "(denoise: one-step nRMSE).", "", "Non-finite test rollouts: " + (", ".join(nonfinite) if nonfinite else "none"),
             "", "Skipped/excluded:"] + [f"- {b}: {why}" for b, why in skipped]
    (HERE / "final_table.md").write_text("\n".join(lines + rk + notes) + "\n")
    print("\n".join(lines + rk + notes))


if __name__ == "__main__":
    main()
