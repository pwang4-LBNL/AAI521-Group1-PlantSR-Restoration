"""Score inpainting methods on the fixed validation/test masks, exactly as the app runs them.

Each method is a TaskHandler (OpenCV baseline, pretrained SD 1.5 or SD 1.5 + a LoRA checkpoint). Inputs come
from BenchmarkExamples: the 256 x 256 evaluation crop with the fixed Week 1 mask, missing pixels filled with
grey. Outputs go through the handler's own `run` (generation + compositing), then `inpaint_metrics` on the
missing region and its boundary. Rows use the team's long results format (`result_rows`).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.data.inpaint_masks import coverage_bin
from src.evaluation.inpaint_metrics import inpaint_metrics, result_rows

PRIMARY_METRIC = "masked_bbox_lpips"  # Week 1 selection metric (lower is better)
LOWER_IS_BETTER = {"masked_bbox_lpips", "masked_l1", "boundary_l1", "boundary_grad_l1"}
REPORT_METRICS = ["masked_bbox_lpips", "masked_psnr", "masked_ssim", "masked_l1", "boundary_l1",
                  "boundary_grad_l1", "full_psnr_supplementary"]


def evaluate(handler, examples, ids, method: str | None = None, lpips_fn=None, seed: int = 42,
             keep_outputs: int = 0, progress=None):
    """Run one method on `ids`; returns (long-format rows, {sample_id: output} for the first `keep_outputs`)."""
    method = method or handler.method
    if not handler.loaded:  # the app's registry does this before the first request
        handler.load()
    rows, outputs = [], {}
    for index, sid in enumerate(ids, start=1):
        image, mask = examples.degraded(sid, "inpainting")
        start = time.perf_counter()
        try:
            out, _ = handler.run(image, mask, seed)
            out = np.asarray(out.convert("RGB") if hasattr(out, "convert") else out)
            seconds = time.perf_counter() - start
            scores = inpaint_metrics(out, examples.reference(sid), mask, lpips_fn=lpips_fn)
            rows += result_rows(sid, method, scores, checkpoint_revision=getattr(handler, "version", ""),
                                seed=seed, inference_seconds=seconds)
            if len(outputs) < keep_outputs:
                outputs[sid] = out
        except Exception as exc:  # noqa: BLE001 - count failures instead of dropping them
            rows += result_rows(sid, method, None, checkpoint_revision=getattr(handler, "version", ""), seed=seed,
                                inference_seconds=time.perf_counter() - start, status="error",
                                error_note=f"{type(exc).__name__}: {exc}")
        if progress and (index % 25 == 0 or index == len(ids)):
            progress(f"{method}: {index}/{len(ids)}")
    return rows, outputs


def wide(rows) -> pd.DataFrame:
    """One row per (method, sample) with one column per metric; failures kept with NaN scores."""
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"].copy()
    ok["metric_value"] = ok["metric_value"].astype(float)
    table = ok.pivot_table(index=["method", "sample_id"], columns="metric_name", values="metric_value")
    failed = df[df.status != "ok"][["method", "sample_id"]].drop_duplicates()
    if len(failed):
        table = pd.concat([table, failed.set_index(["method", "sample_id"])])
    return table.reset_index()


def add_mask_info(table: pd.DataFrame, examples) -> pd.DataFrame:
    info = {sid: examples.mask_info[sid] for sid in table.sample_id.unique()}
    table = table.copy()
    table["mask_type"] = table.sample_id.map(lambda s: info[s]["mask_type"])
    table["coverage_bin"] = table.sample_id.map(lambda s: coverage_bin(info[s]["mask_coverage"]))
    return table


def summary(table: pd.DataFrame, by=("method",)) -> pd.DataFrame:
    metrics = [m for m in REPORT_METRICS if m in table.columns]
    grouped = table.groupby(list(by))
    out = grouped[metrics].mean()
    out["crops"] = grouped.sample_id.nunique()
    out["failures"] = grouped[metrics[0]].apply(lambda s: int(s.isna().sum()))
    return out


def paired(table: pd.DataFrame, a: str, b: str, metrics=None) -> pd.DataFrame:
    """Per-image comparison of method `a` against method `b`: mean difference, share of wins, Wilcoxon p."""
    from scipy.stats import wilcoxon

    metrics = metrics or [m for m in REPORT_METRICS if m in table.columns]
    left = table[table.method == a].set_index("sample_id")
    right = table[table.method == b].set_index("sample_id")
    shared = left.index.intersection(right.index)
    rows = []
    for metric in metrics:
        x, y = left.loc[shared, metric].astype(float), right.loc[shared, metric].astype(float)
        keep = x.notna() & y.notna()
        diff = (x - y)[keep]
        better = diff < 0 if metric in LOWER_IS_BETTER else diff > 0
        try:
            enough = keep.sum() >= 6 and diff.abs().sum() > 0
            p_value = float(wilcoxon(x[keep], y[keep]).pvalue) if enough else float("nan")
        except ValueError:
            p_value = float("nan")
        rows.append({"metric": metric, "better_if": "lower" if metric in LOWER_IS_BETTER else "higher",
                     f"mean_{a}": float(x[keep].mean()), f"mean_{b}": float(y[keep].mean()),
                     "mean_difference": float(diff.mean()), f"{a}_better_share": float(better.mean()),
                     "pairs": int(keep.sum()), "wilcoxon_p": p_value})
    return pd.DataFrame(rows)


def stratified_ids(examples, n: int) -> list[str]:
    """n validation crops spread over mask type and coverage (every k-th crop when sorted by them)."""
    ordered = sorted(examples.ids(), key=lambda s: (examples.mask_info[s]["mask_type"],
                                                    examples.mask_info[s]["mask_coverage"]))
    if n >= len(ordered):
        return ordered
    step = len(ordered) / n
    return [ordered[int(i * step + step / 2)] for i in range(n)]
