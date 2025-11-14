#!/usr/bin/env bash

set -euo pipefail

# Compatible with bash 3.2+ (macOS default).

DATASET="${1:-livejournal}"
ROUNDS="${2:-3}"

BAZEL_FLAGS=(--macos_minimum_os=11.0 --cxxopt=-UPARLAY_USE_STD_ALLOC)

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAP_DIR="${WORKSPACE_DIR}/snap_inputs"
TIMING_DIR="${WORKSPACE_DIR}/timings"

declare SNAP_URL SNAP_FILE SNAP_CONVERTER_FLAGS SNAP_IS_DIRECTED RAW_PATH
NEEDS_CONVERSION=true
NEEDS_COMPRESSED=true

case "${DATASET}" in
  livejournal|soc-LiveJournal1)
    SNAP_URL="https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
    SNAP_FILE="soc-LiveJournal1.txt"
    SNAP_CONVERTER_FLAGS=(--symmetric)
    SNAP_IS_DIRECTED=false
    RAW_PATH="${SNAP_DIR}/${SNAP_FILE}"
    ;;
  twitter|twitter2010|twitter_rv)
    SNAP_URL="https://snap.stanford.edu/data/twitter.tar.gz"
    SNAP_FILE="twitter_rv.net"
    SNAP_CONVERTER_FLAGS=(--symmetric)
    SNAP_IS_DIRECTED=false
    RAW_PATH="${SNAP_DIR}/${SNAP_FILE}"
    ;;
  friendster)
    SNAP_URL="https://snap.stanford.edu/data/com-Friendster.txt.gz"
    SNAP_FILE="com-Friendster.txt"
    SNAP_CONVERTER_FLAGS=(--symmetric)
    SNAP_IS_DIRECTED=false
    RAW_PATH="${SNAP_DIR}/${SNAP_FILE}"
    ;;
  toy_triangles|triangles)
    SNAP_URL=""
    SNAP_FILE="triangles"
    SNAP_CONVERTER_FLAGS=(--symmetric)
    SNAP_IS_DIRECTED=false
    RAW_PATH="${WORKSPACE_DIR}/inputs/triangles.txt"
    NEEDS_CONVERSION=false
    NEEDS_COMPRESSED=false
    ;;
  toy_star|star)
    SNAP_URL=""
    SNAP_FILE="star"
    SNAP_CONVERTER_FLAGS=(--symmetric)
    SNAP_IS_DIRECTED=false
    RAW_PATH="${WORKSPACE_DIR}/inputs/star.txt"
    NEEDS_CONVERSION=false
    NEEDS_COMPRESSED=false
    ;;
  wiki-vote|wiki_vote)
    SNAP_URL="https://snap.stanford.edu/data/wiki-Vote.txt.gz"
    SNAP_FILE="wiki-Vote.txt"
    SNAP_CONVERTER_FLAGS=(--symmetric)
    SNAP_IS_DIRECTED=false
    RAW_PATH="${SNAP_DIR}/${SNAP_FILE}"
    NEEDS_COMPRESSED=false
    ;;
  ego-facebook|facebook|facebook_combined)
    SNAP_URL="https://snap.stanford.edu/data/facebook_combined.txt.gz"
    SNAP_FILE="facebook_combined.txt"
    SNAP_CONVERTER_FLAGS=(--symmetric)
    SNAP_IS_DIRECTED=false
    RAW_PATH="${SNAP_DIR}/${SNAP_FILE}"
    NEEDS_COMPRESSED=false
    ;;
  epinions|soc-epinions1)
    SNAP_URL="https://snap.stanford.edu/data/soc-Epinions1.txt.gz"
    SNAP_FILE="soc-Epinions1.txt"
    SNAP_CONVERTER_FLAGS=(--symmetric)
    SNAP_IS_DIRECTED=false
    RAW_PATH="${SNAP_DIR}/${SNAP_FILE}"
    NEEDS_COMPRESSED=false
    ;;
  ego-twitter|twitter_combined)
    SNAP_URL="https://snap.stanford.edu/data/twitter_combined.txt.gz"
    SNAP_FILE="twitter_combined.txt"
    SNAP_CONVERTER_FLAGS=(--symmetric)
    SNAP_IS_DIRECTED=false
    RAW_PATH="${SNAP_DIR}/${SNAP_FILE}"
    NEEDS_COMPRESSED=false
    ;;
  github|musae-github|github_social)
    SNAP_URL="https://snap.stanford.edu/data/git_web_ml.zip"
    SNAP_FILE="github/musae_git_edges.csv"
    SNAP_CONVERTER_FLAGS=(--symmetric)
    SNAP_IS_DIRECTED=false
    RAW_PATH="${SNAP_DIR}/${SNAP_FILE}"
    NEEDS_COMPRESSED=false
    CONVERT_FROM_CSV=true
    ;;
  *)
    echo "Unsupported dataset '${DATASET}'." >&2
    echo "Supported datasets: livejournal, twitter, friendster, ego-facebook, epinions, ego-twitter, github, toy_triangles, toy_star." >&2
    exit 1
    ;;
esac

mkdir -p "${SNAP_DIR}" "${TIMING_DIR}"

if [[ -z "${RAW_PATH}" ]]; then
  RAW_PATH="${SNAP_DIR}/${SNAP_FILE}"
fi

if [[ -n "${SNAP_URL}" ]]; then
  if [[ "${SNAP_URL}" == *.tar.gz ]]; then
    TAR_ARCHIVE="${SNAP_DIR}/$(basename "${SNAP_URL}")"
    if [[ ! -f "${TAR_ARCHIVE}" ]]; then
      echo "Downloading ${SNAP_URL}"
      curl -L "${SNAP_URL}" -o "${TAR_ARCHIVE}"
    else
      echo "Found existing archive ${TAR_ARCHIVE}"
    fi
    if [[ ! -f "${RAW_PATH}" ]]; then
      echo "Extracting ${TAR_ARCHIVE}"
      tar -xzf "${TAR_ARCHIVE}" -C "${SNAP_DIR}" "${SNAP_FILE}"
    fi
  elif [[ "${SNAP_URL}" == *.gz ]]; then
    GZ_ARCHIVE="${SNAP_DIR}/$(basename "${SNAP_URL}")"
    if [[ ! -f "${GZ_ARCHIVE}" ]]; then
      echo "Downloading ${SNAP_URL}"
      curl -L "${SNAP_URL}" -o "${GZ_ARCHIVE}"
    else
      echo "Found existing archive ${GZ_ARCHIVE}"
    fi
    if [[ ! -f "${RAW_PATH}" ]]; then
      echo "Decompressing ${GZ_ARCHIVE}"
      gunzip -c "${GZ_ARCHIVE}" > "${RAW_PATH}"
    fi
  else
    if [[ ! -f "${RAW_PATH}" ]]; then
      echo "Downloading ${SNAP_URL}"
      curl -L "${SNAP_URL}" -o "${RAW_PATH}"
    fi
  fi
else
  echo "Using local graph ${RAW_PATH}"
fi

if [[ ! -f "${RAW_PATH}" ]]; then
  echo "Failed to materialize raw SNAP graph at ${RAW_PATH}" >&2
  exit 1
fi

if [[ "${CONVERT_FROM_CSV:-false}" == true ]]; then
  CSV_SOURCE="${RAW_PATH}"
  GRAPH_NAME="$(basename "${CSV_SOURCE}" .csv)"
  EDGE_LIST_TXT="${SNAP_DIR}/${GRAPH_NAME}.txt"
  if [[ ! -f "${EDGE_LIST_TXT}" ]]; then
    echo "Converting CSV edge list -> ${EDGE_LIST_TXT}"
    tail -n +2 "${CSV_SOURCE}" | tr ',' ' ' > "${EDGE_LIST_TXT}"
  fi
  RAW_PATH="${EDGE_LIST_TXT}"
fi

GRAPH_BASENAME="$(basename "${RAW_PATH}" .txt)"
GRAPH_BASENAME="${GRAPH_BASENAME%.net}"

if [[ "${NEEDS_CONVERSION}" == true ]]; then
  ADJ_GRAPH="${SNAP_DIR}/${GRAPH_BASENAME}.adj"
  COMPRESSED_GRAPH="${SNAP_DIR}/${GRAPH_BASENAME}.bp"

  if bazel query --noshow_progress //utils:compressor >/dev/null 2>&1; then
    HAS_COMPRESSOR=true
  else
    HAS_COMPRESSOR=false
    NEEDS_COMPRESSED=false
    echo "Warning: //utils:compressor not found; skipping compressed graph generation."
  fi

  echo "Building converters..."
  build_targets=(//utils:snap_converter)
  if [[ "${NEEDS_COMPRESSED}" == true && "${HAS_COMPRESSOR}" == true ]]; then
    build_targets+=(//utils:compressor)
  fi
  bazel build "${BAZEL_FLAGS[@]}" "${build_targets[@]}"

  if [[ ! -f "${ADJ_GRAPH}" ]]; then
    echo "Converting to adjacency format -> ${ADJ_GRAPH}"
    ./bazel-bin/utils/snap_converter \
      "-s" \
      -i "${RAW_PATH}" \
      -o "${ADJ_GRAPH}"
  fi

  if [[ "${NEEDS_COMPRESSED}" == true && "${HAS_COMPRESSOR}" == true && ! -f "${COMPRESSED_GRAPH}" ]]; then
    echo "Generating compressed byte-PDA graph -> ${COMPRESSED_GRAPH}"
    ./bazel-bin/utils/compressor \
      -s \
      -o "${COMPRESSED_GRAPH}" \
      "${ADJ_GRAPH}"
  fi
else
  ADJ_GRAPH="${RAW_PATH}"
  COMPRESSED_GRAPH=""
fi

echo "Building connectivity benchmarks..."
bazel build "${BAZEL_FLAGS[@]}" \
  //benchmarks/Connectivity/WorkEfficientSDB14:Connectivity_main \
  //benchmarks/Connectivity/Gazit1991:Connectivity_main \
  //benchmarks/Connectivity/Gazit1991:Comparison_main

WORK_LOG_DIR="${TIMING_DIR}/${GRAPH_BASENAME}/workefficient"
GAZIT_LOG_DIR="${TIMING_DIR}/${GRAPH_BASENAME}/gazit"
mkdir -p "${WORK_LOG_DIR}" "${GAZIT_LOG_DIR}"

run_and_log() {
  local binary_path="$1"
  local log_path="$2"
  shift 2

  echo "Running ${binary_path} $*" | tee "${log_path}"
  /usr/bin/env time -p "${binary_path}" "$@" 2>&1 | tee -a "${log_path}"
  echo >> "${log_path}"
}

compare_outputs() {
  local graph_path="$1"
  shift
  local extra_flags=($@)

  "${WORKSPACE_DIR}/bazel-bin/benchmarks/Connectivity/Gazit1991/Comparison_main" \
    "${extra_flags[@]}" "${graph_path}" >/dev/null
}

COMMON_FLAGS=(-s -rounds "${ROUNDS}")
if [[ -f "${ADJ_GRAPH}" ]]; then
  compare_outputs "${ADJ_GRAPH}" "${COMMON_FLAGS[@]}"
  run_and_log \
    "${WORKSPACE_DIR}/bazel-bin/benchmarks/Connectivity/WorkEfficientSDB14/Connectivity_main" \
    "${WORK_LOG_DIR}/adjacency.log" \
    "${COMMON_FLAGS[@]}" "${ADJ_GRAPH}"

  run_and_log \
    "${WORKSPACE_DIR}/bazel-bin/benchmarks/Connectivity/Gazit1991/Connectivity_main" \
    "${GAZIT_LOG_DIR}/adjacency.log" \
    "${COMMON_FLAGS[@]}" "${ADJ_GRAPH}"
fi

if [[ -f "${COMPRESSED_GRAPH}" ]]; then
  compare_outputs "${COMPRESSED_GRAPH}" -s -rounds 1 -c -m

  run_and_log \
    "${WORKSPACE_DIR}/bazel-bin/benchmarks/Connectivity/WorkEfficientSDB14/Connectivity_main" \
    "${WORK_LOG_DIR}/compressed.log" \
    -s -c -m -rounds "${ROUNDS}" "${COMPRESSED_GRAPH}"

  run_and_log \
    "${WORKSPACE_DIR}/bazel-bin/benchmarks/Connectivity/Gazit1991/Connectivity_main" \
    "${GAZIT_LOG_DIR}/compressed.log" \
    -s -c -m -rounds "${ROUNDS}" "${COMPRESSED_GRAPH}"
fi

echo "Logs written under ${TIMING_DIR}/${GRAPH_BASENAME}"
