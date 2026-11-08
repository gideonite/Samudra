"""Phase 0 lesion study on a saved model: val metric for all 16 skip masks and for each of the 10 blocks removed.

Usage: python phase0.py runs/<tag> [--split val]   (needs runs/<tag>/model.pt + args.json written by train.py)
Writes runs/<tag>/phase0.json and prints a table. Skip bit order: bit i set = skip i kept (15 = all-keep).
"""
import argparse, json
from pathlib import Path

import torch

import prepare
import train


def lesion(model, name, split, skip_mask=None, block_mask=None):
    with torch.no_grad():
        ev = prepare.evaluate(lambda x: model(x, skip_mask=skip_mask, block_mask=block_mask), name, split=split)
    return dict(metric=ev["metric"], onestep_mse=ev["onestep_mse"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir"); p.add_argument("--split", default="val")
    a = p.parse_args()
    run = Path(a.run_dir); cfg = json.loads((run / "args.json").read_text())
    torch.set_num_threads(1)
    meta = prepare.load(cfg["dataset"])["meta"]
    model = train.build_model(meta, cfg["width"], cfg["drop_mode"]); model.load_state_dict(torch.load(run / "model.pt")); model.eval()
    name = cfg["dataset"]
    res = {"full": lesion(model, name, a.split), "skip_masks": {}, "blocks_removed": {}}
    for m in range(16):
        res["skip_masks"][m] = lesion(model, name, a.split, skip_mask=train.mask_bools(m))
    for b in range(train.N_BLOCKS):
        res["blocks_removed"][b] = lesion(model, name, a.split, block_mask=[i != b for i in range(train.N_BLOCKS)])
    (run / "phase0.json").write_text(json.dumps(res, indent=1))
    f = res["full"]["metric"]
    print(f"full: metric={f:.5g} onestep={res['full']['onestep_mse']:.5g}")
    for m, r in res["skip_masks"].items():
        print(f"skip mask {m:2d} ({m:04b}) metric={r['metric']:.5g} (x{r['metric'] / f:.2f}) onestep={r['onestep_mse']:.5g}")
    for b, r in res["blocks_removed"].items():
        print(f"block {b} removed   metric={r['metric']:.5g} (x{r['metric'] / f:.2f}) onestep={r['onestep_mse']:.5g}")


if __name__ == "__main__":
    main()
