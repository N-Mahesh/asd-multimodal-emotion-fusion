#!/usr/bin/env bash
set -euo pipefail
cd /opt/OpenFace   # ensure models resolve relative

# vars
INPUT_ROOT="/home/openface-build/FinalFullVideos_subjects"
OUTPUT_ROOT="./openface_outputs"
FEATURE_BIN_DEFAULT="$(command -v FeatureExtraction || true)"
FEATURE_BIN="${FEATURE_BIN_DEFAULT:-/home/openface-build/OpenFace/build/bin/FeatureExtraction}"
DRY_RUN=0
JOBS=1

# Running
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--root) INPUT_ROOT="$2"; shift 2;;
    -o|--out)  OUTPUT_ROOT="$2"; shift 2;;
    -b|--bin)  FEATURE_BIN="$2"; shift 2;;
    -j)        JOBS="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# Errors don't quit entire system
[[ -d "$INPUT_ROOT" ]] || { echo "ERROR: INPUT_ROOT not found: $INPUT_ROOT" >&2; exit 1; }
[[ -x "$FEATURE_BIN" ]] || { echo "ERROR: FeatureExtraction not executable: $FEATURE_BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

process_one() {
  local f="$1"

  # Paths
  local rel rel_dir base stem out_dir align_dir hog_dir main_csv aus_csv
  rel="$(realpath --relative-to="$INPUT_ROOT" "$f")"
  rel_dir="$(dirname "$rel")"
  base="$(basename "$f")"
  stem="${base%.*}"
  out_dir="$OUTPUT_ROOT/$rel_dir"
  align_dir="$out_dir/aligned_imgs"
  hog_dir="$out_dir/aligned_hog"
  main_csv="$out_dir/${stem}.csv"
  aus_csv="$out_dir/${stem}_aus.csv"

  if [[ -f "$main_csv" && -f "$aus_csv" ]]; then
    echo "SKIP: $rel (already has ${stem}.csv & ${stem}_aus.csv)"
    return 0
  fi

  mkdir -p "$out_dir" "$align_dir" "$hog_dir"

  local cmd=(
    "$FEATURE_BIN" -q
    -f "$f"
    -out_dir "$out_dir"
    -aus -2Dfp -3Dfp -pdmparams -pose -gaze
    -simalign "$align_dir"
    -hogalign "$hog_dir"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: ${cmd[*]}"
  else
    echo "RUN    : $rel"
    "${cmd[@]}"
  fi
}

export -f process_one
export INPUT_ROOT OUTPUT_ROOT FEATURE_BIN DRY_RUN

# get files
mapfile -d '' FILES < <(
  find "$INPUT_ROOT" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.mpg' -o -iname '*.m4v' \) -print0
)
((${#FILES[@]})) || { echo "No video files under: $INPUT_ROOT" >&2; exit 1; }

echo "Discovered ${#FILES[@]} video(s). Output root: $OUTPUT_ROOT"
echo "FeatureExtraction: $FEATURE_BIN"
[[ "$DRY_RUN" -eq 1 ]] && echo "Mode: DRY RUN"
[[ "$JOBS" -gt 1 ]] && echo "Parallel jobs: $JOBS"

# Ending
printf '%s\0' "${FILES[@]}" \
| xargs -0 -n1 -P "$JOBS" bash -c 'set -euo pipefail; process_one "$1"' _

echo "Done."