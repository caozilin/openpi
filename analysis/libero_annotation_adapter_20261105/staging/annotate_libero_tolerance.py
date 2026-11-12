"""Annotate selected LIBERO nominal episodes with the current Bayesian DSP search."""

import argparse
import signal
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import traceback

from tasktol.data.libero_annotation.source import discover, write
from tasktol.data.libero_annotation.workflow import annotate_episode


def run_job(job):
    source, options = job
    try:
        return annotate_episode(source, **options)
    except Exception as error:
        output = Path(options["output_root"]) / source.suite / f"task_{source.task_id:02d}" / source.demo
        record = dict(
            complete=False,
            status="adapter_error",
            suite=source.suite,
            task_id=source.task_id,
            demo=source.demo,
            source_episode_index=source.episode_index,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        write(output / "adapter_error.json", record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--search-prepared", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--extra",
        type=int,
        default=16,
        help="Adaptive DSP queries per subtask after unchanged structured initialization",
    )
    parser.add_argument("--preflight", action="store_true", help="Only first two structured probes; no labels")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--mask-table", type=Path)
    parser.add_argument("--robosuite-assets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--episode", action="append", help="Exact suite/task_id/demo_id; repeat to select a small sample"
    )
    parser.add_argument("--all", action="store_true", help="Explicitly select all nominal episodes")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve source episodes only, without loading models or searching"
    )
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    if args.search_prepared:
        from tasktol.data.libero_annotation.dsp import search_prepared

        print(search_prepared(args.search_prepared, extra=args.extra, preflight=args.preflight), flush=True)
        return
    if args.openpi_root is None or args.output is None:
        parser.error("--openpi-root and --output are required")
    root = args.openpi_root.resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if bool(args.episode) == args.all:
        parser.error("Select explicit --episode entries or --all")
    raw = args.raw_dir or root / "data/libero_tasktol_joint_servo/20261101_full_v1"
    index = (
        args.source_index
        or root / "data/lerobot/caozilin/libero_reconstructed_full_20261101/meta/reconstruction_sources.json"
    )
    masks = args.mask_table or root / "examples/libero/configs/libero40_semantic_tolerances_deg.json"
    sources = discover(raw, index)
    if not args.all:
        requested = set(args.episode)
        if len(requested) != len(args.episode):
            parser.error("Duplicate requested episodes")
        sources = tuple(s for s in sources if f"{s.suite}/{s.task_id:02d}/{s.demo}" in requested)
        if len(sources) != len(requested):
            parser.error("Requested episodes are not in the audited nominal dataset")
    manifest = dict(
        scope="pure offline kinematics; self and table geometry; no physics or policy inference",
        planned_episodes=len(sources),
        control_hz=20,
        workers=args.workers,
        sources=[
            dict(
                episode_index=s.episode_index,
                suite=s.suite,
                task_id=s.task_id,
                demo=s.demo,
                folder=str(s.folder),
                frames=s.frames,
            )
            for s in sources
        ],
    )
    if args.dry_run:
        print(manifest, flush=True)
        return
    write(args.output / "annotation_request.json", manifest)
    options = dict(
        openpi_root=root,
        output_root=args.output,
        mask_table=masks,
        robosuite_assets=args.robosuite_assets,
        extra=args.extra,
        preflight=args.preflight,
    )
    jobs = [(s, options) for s in sources]
    results = []
    if args.workers == 1:
        iterator = map(run_job, jobs)
        for result in iterator:
            results.append(result)
            print(result, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(run_job, jobs):
                results.append(result)
                print(result, flush=True)
    summary = dict(
        complete=all(r["complete"] for r in results),
        processing_complete=all(r["complete"] for r in results),
        all_labeled=all(r["status"] == "labeled" for r in results),
        processed=len(results),
        labeled=sum(r["status"] == "labeled" for r in results),
        no_feasible_label=sum(r["status"] == "no_feasible_label" for r in results),
        adapter_errors=sum(r["status"] == "adapter_error" for r in results),
        results=results,
    )
    write(args.output / "annotation_summary.json", summary)
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
