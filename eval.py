import argparse
import h5py
import json
import numpy as np
import sys


def evaluate_single(result_path, gt_knns, k):
    """Evaluates a single result file against the pre-loaded ground truth."""
    try:
        with h5py.File(result_path, 'r') as f_res:
            # Decode attributes (handles bytes vs strings)
            def decode_attr(attr):
                if isinstance(attr, bytes):
                    return attr.decode('utf-8')
                return str(attr) if attr is not None else "unknown"

            metadata = {
                "file": result_path,
                "k_evaluated": k,
                "build_time_sec": float(f_res.attrs.get('buildtime', 0.0)),
                "query_time_sec": float(f_res.attrs.get('querytime', 0.0)),
                "build_params": decode_attr(f_res.attrs.get('build_params')),
                "search_params": decode_attr(f_res.attrs.get('search_params'))
            }

            # Load Result KNNs
            res_knns = f_res['knns'][:, :k]

            if gt_knns.shape[0] != res_knns.shape[0]:
                print(
                    f"  [!] Error: Mismatch in queries for {result_path}. GT: {gt_knns.shape[0]}, Res: {res_knns.shape[0]}")
                return None

            n_queries = gt_knns.shape[0]
            total_recall = 0.0

            # Compute Recall@K
            for i in range(n_queries):
                gt_set = set(gt_knns[i])
                res_set = set(res_knns[i])
                intersection = len(gt_set.intersection(res_set))
                total_recall += intersection / k

            recall_at_k = total_recall / n_queries
            metadata[f"recall@{k}"] = round(recall_at_k, 4)

            return metadata

    except Exception as e:
        print(f"  [!] Evaluation failed for {result_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate NMF-IVF search results.")
    # nargs='+' allows accepting multiple files from a wildcard like *.h5
    parser.add_argument("--results", nargs='+', required=True, help="Path(s) to your C++ output HDF5 files")
    parser.add_argument("--gt", required=True, help="Path to the original dataset HDF5 with ground truth")
    parser.add_argument("--k", type=int, default=30, help="Calculate Recall at K (default: 30)")
    parser.add_argument("--output", default="metrics_summary.json", help="Path to save the summary JSON metrics")
    parser.add_argument("--target-recall", type=float, default=0.0,
                        help="Minimum recall threshold for finding the 'best' run")

    args = parser.parse_args()

    print(f"Loading Ground Truth from {args.gt}...")
    try:
        with h5py.File(args.gt, 'r') as f_gt:
            gt_knns = f_gt['/otest/knns'][:, :args.k]
    except Exception as e:
        print(f"Fatal Error loading Ground Truth: {e}")
        sys.exit(1)

    all_metrics = []

    print(f"Evaluating {len(args.results)} result file(s)...")
    for res_path in args.results:
        metrics = evaluate_single(res_path, gt_knns, args.k)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        print("No valid results were evaluated. Exiting.")
        sys.exit(1)

    # --- Find the "Best" Run ---
    # Filter runs that meet the minimum target recall
    valid_runs = [m for m in all_metrics if m[f"recall@{args.k}"] >= args.target_recall]

    if valid_runs:
        # If runs met the threshold, the best one is the FASTEST query time among them
        best_run = min(valid_runs, key=lambda x: x["query_time_sec"])
        achieved_target = True
    else:
        # If no run met the threshold, fallback to finding the HIGHEST recall possible
        best_run = max(all_metrics, key=lambda x: (x[f"recall@{args.k}"], -x["query_time_sec"]))
        achieved_target = False

    # Sort all metrics for the JSON output (Highest recall first, tie-break by lowest query time)
    all_metrics.sort(key=lambda x: (x[f"recall@{args.k}"], -x["query_time_sec"]), reverse=True)

    summary = {
        "best_run": best_run,
        "target_recall_achieved": achieved_target,
        "all_runs": all_metrics
    }

    # Save to JSON
    with open(args.output, 'w') as f_out:
        json.dump(summary, f_out, indent=4)

    # --- Console Output ---
    print("\n========================================")
    print(" Evaluation Summary")
    print("========================================")
    print(f" Total files evaluated : {len(all_metrics)}")
    print(f" Output saved to       : {args.output}")
    print("\n -> BEST RUN IDENTIFIED <-")
    if args.target_recall > 0.0 and not achieved_target:
        print(f" (Warning: No run achieved the target recall of {args.target_recall})")

    print(f" File          : {best_run['file']}")
    print(f" Recall@{args.k}    : {best_run[f'recall@{args.k}']:.4f}")
    print(f" Query Time    : {best_run['query_time_sec']:.4f} s")
    print(f" Build Params  : {best_run['build_params']}")
    print(f" Search Params : {best_run['search_params']}")
    print("========================================\n")


if __name__ == "__main__":
    main()
