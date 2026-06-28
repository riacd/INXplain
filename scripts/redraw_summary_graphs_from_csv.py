#!/usr/bin/env python3
"""
Re-draw summary graph visualizations from saved CSV files.

Example:
python scripts/redraw_summary_graphs_from_csv.py \
  --summary-dir results/case_study_karate/KarateClub_original_GCN/summary_graphs \
  --model-name gradient_based
"""

import argparse
import importlib.util
from pathlib import Path

# Load the utility module directly to avoid importing GS/__init__.py
# (which may require training-time dependencies such as torch).
PROJECT_ROOT = Path(__file__).parent.parent
MODULE_PATH = PROJECT_ROOT / "GS" / "utils" / "summary_graph_visualization.py"
SPEC = importlib.util.spec_from_file_location("summary_graph_visualization", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Failed to load visualization module: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
visualize_from_summary_csv = MODULE.visualize_from_summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-draw summary graph visualizations from summary_graphs CSV files.",
    )
    parser.add_argument(
        "--summary-dir",
        required=True,
        help="Path to summary_graphs directory containing *_step_*_edges.csv",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model prefix used in CSV names (e.g., gradient_based). "
             "Required only when multiple models exist in one directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for generated PNG files (default: sibling graph_visualizations directory).",
    )
    parser.add_argument(
        "--node-info",
        default=None,
        help="Path to node_info.csv (default: <summary-dir>/node_info.csv).",
    )
    parser.add_argument(
        "--step-metrics",
        default=None,
        help="Path to *_step_metrics.tsv for adding Accuracy in titles "
             "(default: inferred from sibling process_results directory).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200,
        help="Skip plotting when node count exceeds this threshold (default: 200).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    if not summary_dir.exists():
        print(f"❌ Summary directory not found: {summary_dir}")
        return 1

    try:
        generated = visualize_from_summary_csv(
            summary_graphs_dir=summary_dir,
            model_name=args.model_name,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            node_info_path=Path(args.node_info) if args.node_info else None,
            step_metrics_path=Path(args.step_metrics) if args.step_metrics else None,
            max_nodes=args.max_nodes,
        )
    except Exception as exc:
        print(f"❌ Re-draw failed: {exc}")
        return 1

    if not generated:
        print(
            "⚠️ No figures generated. "
            "Possible reasons: node count exceeds --max-nodes, "
            "or no valid input files were found."
        )
        return 0

    print(f"✅ Re-draw completed. Generated {len(generated)} files.")
    print(f"📁 Output directory: {generated[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
