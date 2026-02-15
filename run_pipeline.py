"""
Run the full Opticube pipeline: data build + optimize.

Steps:
  1. Download Scryfall bulk data (if not fresh)
  2. Download EDHREC synergy data for top N cards (skip fresh local files)
  3. Build sparse matrices from the JSON data
  4. Build card metadata table (join EDHREC + Scryfall)
  5. (Optional) Run optimizer to select best K cards

Usage:
    # Data only
    python run_pipeline.py --top 500 --skip-download

    # Data + optimize (one command does everything)
    python run_pipeline.py --top 500 -k 10 --metric ppmi
    python run_pipeline.py --top 500 -k 30 --metric ppmi --lambda-color 20 --lambda-type 50
    python run_pipeline.py --top 500 -k 10 --skip-download --metric lift

    # Full pipeline from scratch
    python run_pipeline.py --top 5000 -k 360 --metric ppmi --lambda-color 20 --lambda-type 50 --lambda-curve 30

    # Linear objective (auto-selects ILP for small pools)
    python run_pipeline.py --top 500 -k 10 --metric ppmi --objective linear
"""

import argparse
import time

from download_scryfall import download_scryfall_bulk
from scrape_edhrec import download_all as download_edhrec
from load_cards_files import build_matrices
from build_metadata import build_metadata

DEFAULT_TOP_N = 5000
DEFAULT_DELAY = 1.0
DEFAULT_MAX_AGE_DAYS = 14


def run_data_pipeline(
    top_n: int,
    delay: float,
    max_age_days: float,
    skip_download: bool = False,
    force: bool = False,
):
    """Run the data pipeline (steps 1-4)."""
    # Step 1: Scryfall
    print("=" * 60)
    print("STEP 1: Scryfall Bulk Data")
    print("=" * 60)
    if skip_download:
        print("(skipped)")
    else:
        download_scryfall_bulk(max_age_days=max_age_days, force=force)
    print()

    # Step 2: EDHREC
    print("=" * 60)
    print(f"STEP 2: EDHREC Synergy Data (top {top_n})")
    print("=" * 60)
    if skip_download:
        print("(skipped)")
    else:
        download_edhrec(top_n, delay, max_age_days, force=force)
    print()

    # Step 3: Matrices
    print("=" * 60)
    print("STEP 3: Build Matrices")
    print("=" * 60)
    build_matrices(pool_size=top_n)
    print()

    # Step 4: Metadata
    print("=" * 60)
    print("STEP 4: Build Card Metadata")
    print("=" * 60)
    build_metadata()
    print()


def run_optimizer(
    k: int,
    metric: str = 'ppmi',
    objective: str = 'log',
    effort: str = 'normal',
    num_restarts: int = None,
    max_no_improve: int = None,
    lambda_color: float = 0.0,
    lambda_type: float = 0.0,
    lambda_curve: float = 0.0,
    seed: int = 42,
    card_metadata: str = 'data/card_metadata.json',
):
    """Run the optimizer (step 5)."""
    import pickle
    import scipy.sparse as sp
    from rich.console import Console
    from rich.markdown import Markdown
    from optimizer import (
        load_pool_metadata,
        solve_cube,
        display_cube_results,
        resolve_effort,
        _print_solver_report,
    )

    console = Console(width=100, soft_wrap=True)

    # Resolve effort preset
    num_restarts, max_no_improve = resolve_effort(effort, num_restarts, max_no_improve)

    print("=" * 60)
    print(f"STEP 5: Optimize (K={k}, metric={metric}, objective={objective}, "
          f"effort={effort})")
    print("=" * 60)

    # Select matrix based on metric
    if metric == 'ppmi':
        matrix_path = 'incoming_ppmi_matrix.pkl'
    elif metric == 'lift':
        matrix_path = 'incoming_lift_matrix.pkl'
    else:
        matrix_path = 'incoming_synergy_matrix.pkl'

    console.print(Markdown("### Loading data"))
    console.print(f"  Metric: **{metric}** (matrix: {matrix_path})")
    console.print(f"  Objective: **{objective}**")
    console.print(f"  Effort: **{effort}** "
                  f"({num_restarts} restarts, patience {max_no_improve})")

    with open(matrix_path, 'rb') as f:
        incoming_matrix = pickle.load(f)
    console.print(f"  Incoming matrix: {incoming_matrix.shape}, "
                  f"{incoming_matrix.nnz} non-zero")

    with open('card_names.pkl', 'rb') as f:
        card_names = pickle.load(f)
    console.print(f"  Card pool: {len(card_names)} cards (N={len(card_names)})")

    pool_meta = load_pool_metadata(card_names, card_metadata)

    lambdas = {
        'color': lambda_color,
        'type': lambda_type,
        'curve': lambda_curve,
    }
    use_outer_log = (objective == 'log')

    cube_indices, score, details = solve_cube(
        incoming_matrix=incoming_matrix,
        card_names=card_names,
        pool_meta=pool_meta,
        k=k,
        lambdas=lambdas,
        use_outer_log=use_outer_log,
        num_restarts=num_restarts,
        max_no_improve=max_no_improve,
        seed=seed,
    )

    # Solver report
    solver_result = details.get('solver_result')
    if solver_result:
        _print_solver_report(solver_result, effort_name=effort,
                             metric=metric, objective=objective)

    M_dense = incoming_matrix.toarray() if sp.issparse(incoming_matrix) else incoming_matrix
    display_cube_results(
        cube_indices, card_names, pool_meta, score, details, M_dense,
        lambdas, use_outer_log=use_outer_log)

    return cube_indices, score


def main():
    parser = argparse.ArgumentParser(
        description="Opticube — build data and optimize in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Data only
  python run_pipeline.py --top 500 --skip-download

  # One command: build data + optimize (PPMI is default metric)
  python run_pipeline.py --top 500 -k 10
  python run_pipeline.py --top 500 -k 30 --lambda-type 50

  # Full pipeline with balance penalties
  python run_pipeline.py --top 500 -k 10 --metric ppmi \\
      --lambda-color 20 --lambda-type 50 --lambda-curve 30

  # Linear objective (auto-selects ILP for small pools)
  python run_pipeline.py --top 500 -k 10 --objective linear

  # Compare metrics
  python run_pipeline.py --top 500 --skip-download -k 10 --metric ppmi
  python run_pipeline.py --top 500 --skip-download -k 10 --metric lift
        """,
    )

    # --- Data pipeline args ---
    data_group = parser.add_argument_group("Data pipeline")
    data_group.add_argument(
        "--top", type=int, default=DEFAULT_TOP_N,
        help=f"Pool size N: number of top cards by frequency (default: {DEFAULT_TOP_N})"
    )
    data_group.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Delay between EDHREC requests in seconds (default: {DEFAULT_DELAY})"
    )
    data_group.add_argument(
        "--max-age", type=float, default=DEFAULT_MAX_AGE_DAYS,
        help=f"Max age in days before re-downloading (default: {DEFAULT_MAX_AGE_DAYS})"
    )
    data_group.add_argument(
        "--skip-download", action="store_true",
        help="Skip download steps, only rebuild matrices and metadata from local data"
    )
    data_group.add_argument(
        "--force", action="store_true",
        help="Force re-download of everything, ignoring local cache"
    )

    # --- Optimizer args ---
    opt_group = parser.add_argument_group("Optimizer (pass -k to enable)")
    opt_group.add_argument(
        "-k", type=int, default=None,
        help="Cube size K: number of cards to select. If omitted, skip optimization."
    )
    opt_group.add_argument(
        "--metric", choices=["synergy", "lift", "ppmi"], default="ppmi",
        help="EDHREC metric for synergy scoring (default: ppmi)"
    )
    opt_group.add_argument(
        "--objective", choices=["log", "linear"], default="log",
        help="Objective: log = ln(1+incoming), linear = raw sum (default: log)"
    )
    opt_group.add_argument(
        "--effort", choices=["quick", "normal", "thorough", "max"], default="normal",
        help="Effort preset: quick (~2s), normal (~10s), thorough (~1min), max (~5min)"
    )
    opt_group.add_argument(
        "--lambda-color", type=float, default=0.0,
        help="Color balance penalty weight (default: 0.0)"
    )
    opt_group.add_argument(
        "--lambda-type", type=float, default=0.0,
        help="Type balance penalty weight (default: 0.0)"
    )
    opt_group.add_argument(
        "--lambda-curve", type=float, default=0.0,
        help="Mana curve penalty weight (default: 0.0)"
    )
    opt_group.add_argument(
        "--num-restarts", type=int, default=None,
        help="Number of random restarts (overrides --effort)"
    )
    opt_group.add_argument(
        "--max-no-improve", type=int, default=None,
        help="Stop local search after N non-improving swaps (overrides --effort)"
    )
    opt_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    opt_group.add_argument(
        "--card-metadata", type=str, default="data/card_metadata.json",
        help="Path to card_metadata.json (default: data/card_metadata.json)"
    )

    args = parser.parse_args()

    print(f"=== Opticube Pipeline ===")
    print(f"Pool size N: {args.top}")
    if args.k is not None:
        print(f"Cube size K: {args.k}")
        print(f"Metric: {args.metric}")
        print(f"Objective: {args.objective}")
    else:
        print(f"Optimize: NO (pass -k to enable)")
    if args.skip_download:
        print("Skip download: YES")
    if args.force:
        print("Force: YES")
    print()

    overall_start = time.time()

    try:
        # Steps 1-4: data pipeline
        run_data_pipeline(
            top_n=args.top,
            delay=args.delay,
            max_age_days=args.max_age,
            skip_download=args.skip_download,
            force=args.force,
        )

        # Step 5: optimize (if -k is given)
        if args.k is not None:
            run_optimizer(
                k=args.k,
                metric=args.metric,
                objective=args.objective,
                effort=args.effort,
                num_restarts=args.num_restarts,
                max_no_improve=args.max_no_improve,
                lambda_color=args.lambda_color,
                lambda_type=args.lambda_type,
                lambda_curve=args.lambda_curve,
                seed=args.seed,
                card_metadata=args.card_metadata,
            )

        elapsed = time.time() - overall_start
        print()
        print("=" * 60)
        print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Data saved so far is safe.")
        print("Re-run to resume from where you left off.")


if __name__ == "__main__":
    main()
