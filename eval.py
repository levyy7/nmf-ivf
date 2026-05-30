import argparse
import h5py
import json
import numpy as np
import sys


def evaluate(result_path, gt_path, k, output_path):
    print(f"Loading results from {result_path} and GT from {gt_path}...")

    try:
        with h5py.File(result_path, 'r') as f_res, h5py.File(gt_path, 'r') as f_gt:
            # 1. Read Metadata from your C++ attributes
            # Decoding from bytes if h5py returns byte strings
            def decode_attr(attr):
                if isinstance(attr, bytes):
                    return attr.decode('utf-8')
                return str(attr) if attr is not None else "unknown"

            metadata = {
                "dataset": "fiqa-dev",
                "k_evaluated": k,
                "build_time_sec": float(f_res.attrs.get('buildtime', 0.0)),
                "query_time_sec": float(f_res.attrs.get('querytime', 0.0)),
                "build_params": decode_attr(f_res.attrs.get('build_params')),
                "search_params": decode_attr(f_res.attrs.get('search_params'))
            }

            # 2. Load Ground Truth and Result KNNs (Slice to top-k)
            gt_knns = f_gt['/otest/knns'][:, :k]
            res_knns = f_res['knns'][:, :k]

            if gt_knns.shape[0] != res_knns.shape[0]:
                print(f"Error: Mismatch in number of queries! GT: {gt_knns.shape[0]}, Res: {res_knns.shape[0]}")
                sys.exit(1)

            n_queries = gt_knns.shape[0]
            total_recall = 0.0

            # 3. Compute Recall@K
            for i in range(n_queries):
                # Using sets for fast intersection
                gt_set = set(gt_knns[i])
                res_set = set(res_knns[i])

                # How many of the true nearest neighbors did we find?
                intersection = len(gt_set.intersection(res_set))
                total_recall += intersection / k

            recall_at_k = total_recall / n_queries
            metadata[f"recall@{k}"] = round(recall_at_k, 4)

            # 4. Save to JSON
            with open(output_path, 'w') as f_out:
                json.dump(metadata, f_out, indent=4)

            print("========================================")
            print(f" Evaluation Complete")
            print("========================================")
            print(f" Recall@{k}      : {recall_at_k:.4f}")
            print(f" Build Time    : {metadata['build_time_sec']:.2f} s")
            print(f" Query Time    : {metadata['query_time_sec']:.2f} s")
            print(f" Output saved  : {output_path}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NMF-IVF search results.")
    parser.add_argument("--results", required=True, help="Path to your C++ output HDF5")
    parser.add_argument("--gt", required=True, help="Path to the original dataset HDF5 with ground truth")
    parser.add_argument("--k", type=int, default=30, help="Calculate Recall at K (default: 30)")
    parser.add_argument("--output", default="metrics.json", help="Path to save the JSON metrics")

    args = parser.parse_args()
    evaluate(args.results, args.gt, args.k, args.output)
