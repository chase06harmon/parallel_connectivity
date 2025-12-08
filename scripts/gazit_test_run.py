"""
This script runs all the benchmark binaries on specified input graphs.

Everything listed as a Bazel `cc_binary` under `benchmarks/` must be listed in
either `UNWEIGHTED_GRAPH_BENCHMARKS`, `WEIGHTED_GRAPH_BENCHMARKS`, or
`IGNORED_BINARIES` below. Forcing this explicit labeling stops contributors from
forgetting to update this file when they add a new benchmark.

The script only checks that the benchmarks run and exit without an error. It
does not check that the output of each benchmark is correct.

This script could be extended to further split benchmarks into ones that process
symmetric graphs versus asymmetric graphs, but currently the input graphs must
be symmetric so that all benchmarks can run on them.

This script should be invoked directly via Python >=3.7. Because this script
calls other Bazel commands, invoking it with `bazel run` won't work.
"""
from typing import List, Optional, Set, Tuple
import argparse
import fnmatch
import os
import sys
import subprocess

# The script will invoke these benchmark on an unweighted graph.
BENCHMARKS = [
    "//benchmarks/Connectivity/LabelPropagation:Connectivity_main",
    "//benchmarks/Connectivity/SimpleUnionAsync:Connectivity_main",
    "//benchmarks/Connectivity/WorkEfficientSDB14:Connectivity_main",
    "//benchmarks/Connectivity/Gazit1991:Connectivity_main",
]



def get_all_benchmark_binaries() -> List[str]:
    """Returns a list of all binaries under `benchmarks/`."""
    return subprocess.run(
        ["bazel", "query", "kind(cc_binary, //benchmarks/...)"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()


def check_listed_binaries(
    valid_binaries: List[str], ignored_binaries: List[str]
) -> None:
    """Checks listed binaries for consistency.

    Args:
        valid_binaries: Names of binaries that are considered valid.
        ignored_binaries: Names of binaries that are considered invalid
            and should not be run.

    Raises:
        ValueError: `valid_binaries` and `ignored_binaries` overlap.
        ValueError: `valid_binaries` and `ignored_binaries` combined don't equal
            the set of all benchmark-related binaries as determined by
            `get_all_benchmark_binaries()`.
    """
    remaining_binaries = set(get_all_benchmark_binaries())
    for ignored_binaries_pattern in ignored_binaries:
        conflicting_binaries = fnmatch.filter(valid_binaries, ignored_binaries_pattern)
        if conflicting_binaries:
            raise ValueError(
                "Benchmarks listed as both valid and ignored: {}".format(
                    conflicting_binaries
                )
            )
        binaries_to_ignore = fnmatch.filter(
            remaining_binaries, ignored_binaries_pattern
        )
        if not binaries_to_ignore:
            print(
                "Warning: ignore rule {} has no effect".format(ignored_binaries_pattern)
            )
        remaining_binaries -= set(binaries_to_ignore)

    valid_binaries = set(valid_binaries)
    if remaining_binaries != valid_binaries:
        extra_listed_binaries = valid_binaries - remaining_binaries
        if extra_listed_binaries:
            raise ValueError(
                "Listed benchmarks do not exist: {}".format(extra_listed_binaries)
            )
        missing_listed_binaries = remaining_binaries - valid_binaries
        if missing_listed_binaries:
            raise ValueError(
                "Please update {} to include binaries {}".format(
                    __file__, missing_listed_binaries
                )
            )
        

def filter_binaries(
    valid_binaries: List[str]
) -> List[str]:
    """Checks listed binaries for consistency.

    Args:
        valid_binaries: Names of binaries that are considered valid.
        ignored_binaries: Names of binaries that are considered invalid
            and should not be run.

    Raises:
        ValueError: `valid_binaries` and `ignored_binaries` overlap.
        ValueError: `valid_binaries` and `ignored_binaries` combined don't equal
            the set of all benchmark-related binaries as determined by
            `get_all_benchmark_binaries()`.
    """
    remaining_binaries = set(get_all_benchmark_binaries())
    valid_binaries = set(valid_binaries)

    if remaining_binaries != valid_binaries:
        return list(remaining_binaries & valid_binaries)


def run_all_benchmarks(
    unweighted_graph_benchmarks: List[str],
    weighted_graph_benchmarks: List[str],
    unweighted_graph_file: Optional[str],
    weighted_graph_file: Optional[str],
    are_graphs_compressed: bool,
    timeout: Optional[int],
    rounds: Optional[int],
) -> List[Tuple[str, str]]:
    """Runs all benchmarks, returning a list of failing benchmarks.

    Args:
        unweighted_graph_benchmarks: List of all benchmarks to run on the
            unweighted graph.
        weighted_graph_benchmarks: List of all benchmarks to run on the
            weighted graph.
        unweighted_graph_file: File path to the unweighted graph.
        weighted_graph_file: File path to the weighted graph.
        are_compressed_compressed: Whether the graph files hold compressed
            graphs.
        timeout: Benchmarks that run longer than this timeout period in seconds
            are considered to have failed. If this is `None` then the benchmarks
            have no time limit.

    Returns:
        A list of names of benchmarks that fail along with a failure reason.
    """

    BAZEL_FLAGS = ["--compilation_mode", "opt"]
    gbbs_flags = ["-s", "-rounds", rounds]
    if are_graphs_compressed:
        gbbs_flags += ["-c"]

    benchmarks = []
    if unweighted_graph_file:
        benchmarks += unweighted_graph_benchmarks
    if weighted_graph_file:
        benchmarks += weighted_graph_benchmarks
    # Compile all the benchmarks up front --- it's faster than compiling
    # them individually since Bazel can compile several files in parallel.
    subprocess.run(["bazel", "build"] + BAZEL_FLAGS + ["--keep_going"] + benchmarks)

    failed_benchmarks = []

    def test_benchmark(
        benchmark: str, graph_file: str, additional_gbbs_flags: List[str]
    ) -> None:
        try:
            benchmark_run = subprocess.run(
                ["bazel", "run"]
                + BAZEL_FLAGS
                + [benchmark, "--"]
                + gbbs_flags
                + additional_gbbs_flags
                + [graph_file],
                timeout=timeout,
            )
            if benchmark_run.returncode:
                failed_benchmarks.append(
                    (
                        benchmark,
                        "Exited with error code {}".format(benchmark_run.returncode),
                    )
                )
        except subprocess.TimeoutExpired:
            failed_benchmarks.append((benchmark, "Timeout"))

    if unweighted_graph_file:
        for benchmark in unweighted_graph_benchmarks:
            print(f"{"-"*30}\nStart {benchmark}\n")
            test_benchmark(
                benchmark=benchmark,
                graph_file=unweighted_graph_file,
                additional_gbbs_flags=[],
            )
            print(f"\nEnd {benchmark}\n{"-"*30}\n")
    if weighted_graph_file:
        for benchmark in weighted_graph_benchmarks:
            test_benchmark(
                benchmark=benchmark,
                graph_file=weighted_graph_file,
                additional_gbbs_flags=["-w"],
            )

    return failed_benchmarks


if __name__ == "__main__":
    binaries_to_run = filter_binaries(BENCHMARKS)

    parser = argparse.ArgumentParser(
        description=(
            "Runs all benchmarks on the specified input graphs to check "
            "whether the benchmarks run and exit without errors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--unweighted_graph",
        "-u",
        type=str,
        help=(
            "Absolute path to an unweighted graph on which to run all "
            "unweighted graph benchmarks. If not provided, those benchmarks "
            "will not be run."
        ),
    )
    parser.add_argument(
        "--weighted_graph",
        "-w",
        type=str,
        help=(
            "Absolute path to a weighted graph on which to run all "
            "weighted graph benchmarks. If not provided, those benchmarks will "
            "not be run."
        ),
    )
    parser.add_argument(
        "--compressed",
        "-c",
        action="store_true",
        help="Add this flag if input graphs are compressed graphs.",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=60,
        help="(seconds) - Halt benchmarks that run longer than this time.",
    )

    parser.add_argument(
        "--rounds",
        "-r",
        type=float,
        default=1,
        help="rounds per benchmark.",
    )

    parsed_args = parser.parse_args()
    if not parsed_args.unweighted_graph and not parsed_args.weighted_graph:
        parser.error(
            "At least one of --unweighted_graph and --weighted_graph is required."
        )

    unweighted_graph_file = (
        os.path.abspath(parsed_args.unweighted_graph)
        if parsed_args.unweighted_graph
        else None
    )
    weighted_graph_file = (
        os.path.abspath(parsed_args.weighted_graph)
        if parsed_args.weighted_graph
        else None
    )

    failed_benchmarks = run_all_benchmarks(
        unweighted_graph_benchmarks=binaries_to_run,
        weighted_graph_benchmarks=[],
        unweighted_graph_file=unweighted_graph_file,
        weighted_graph_file=weighted_graph_file,
        are_graphs_compressed=parsed_args.compressed,
        timeout=parsed_args.timeout,
        rounds = parsed_args.rounds,
    )
    if failed_benchmarks:
        print("Benchmarks failed: {}".format(failed_benchmarks))
        sys.exit(1)
    else:
        print("Success! All benchmarks completed without an error.")
        sys.exit(0)
