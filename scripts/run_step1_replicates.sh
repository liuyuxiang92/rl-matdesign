#!/usr/bin/env bash
# Run from your RL environment with rl-matdesign installed.
# YAML paths may be absolute, checkout-relative, or caller-relative.
# Bare filenames also resolve under the selected checkout's configs/ directory.
# Submit independent PopArt and raw-reward jobs to a detached local queue.
set -euo pipefail

# Take a private copy of the scenario settings for each run. This also works
# with installed CLIs that expose the generation count only through YAML.
if [[ ${1:-} == --run-one || ${1:-} == --check-one ]]; then
    job_action=$1
    config=$2
    method=$3
    out=$4
    shift 4
    if [[ $job_action == --check-one ]]; then
        effective=$(mktemp)
        trap 'rm -f -- "$effective"' EXIT
    else
        effective="$out/generation_config.yaml"
    fi
    python_bin=$(command -v python || command -v python3)
    "$python_bin" - "$config" "$effective" "$method" <<'PY'
import os
import sys
from pathlib import Path

import yaml
from rl_matdesign.config_schema import load_yaml_config, select_method_values, validate_config

source, destination, method = sys.argv[1:]
cfg = load_yaml_config(source)
validate_config(select_method_values(cfg, method=method))
cfg["num_gen_eps"] = 1000
cfg["max_gen_attempts"] = 100000
# Preserve scenario exploration settings and method maps. Omitted targets use
# the runtime defaults; retired fields remain errors, never automatic migrations.
for prop in cfg["properties"]:
    if prop.get("predictor") == "rf_magpie":
        override = os.environ.get("SINTER_MODEL") if prop.get("name") == "sinter" else None
        if override:
            prop["models"] = [str(Path(override).expanduser().resolve())]
        for model in prop.get("models", []):
            if not Path(model).is_file():
                sys.exit(f"Missing predictor {Path(model).absolute()}. Check the YAML model path or set SINTER_MODEL.")
Path(destination).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
    if [[ $job_action == --check-one ]]; then
        "${RL_MATDESIGN_CLI:-rl-matdesign}" --config "$effective" "$@" --validate-config
        exit
    fi
    "${RL_MATDESIGN_CLI:-rl-matdesign}" --config "$effective" "$@" --out "$out"
    "$python_bin" - "$out/generated.csv" <<'PY'
import csv
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
keys = set()
rows = 0
if path.is_file():
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (json.dumps(json.loads(row["candidate_json"]), sort_keys=True)
                   if row.get("candidate_json") else row["formula"].strip())
            if key:
                keys.add(key)
            rows += 1
if rows != 1000 or len(keys) != 1000:
    sys.exit(f"Generation incomplete: {len(keys)}/1000 distinct candidates in {rows} rows. See {path}.")
print("Verified 1000 distinct generated candidates.")
PY
    exit
fi

action=submit
existing_batch=
output_dir=
trained_batch=
methods=(a2c dqn-mc dqn-bootstrap)
modes=(popart none)
seeds=(11 29 47)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --validate) action=validate ;;
        --dry-run) action=preview ;;
        --include-bootstrap) ;;  # Already included by default.
        --reuse-trained)
            [[ $# -ge 2 && -d "$2" ]] || { echo "--reuse-trained needs the previous batch directory." >&2; exit 2; }
            trained_batch=$(cd -- "$2" && pwd -P)
            shift ;;
        --out)
            [[ -n ${2:-} ]] || { echo "--out needs a results directory." >&2; exit 2; }
            output_dir=$2
            shift ;;
        --continue-batch)
            [[ $# -ge 2 && -d "$2" ]] || { echo "--continue-batch needs an existing batch directory." >&2; exit 2; }
            existing_batch=$(cd -- "$2" && pwd -P)
            shift ;;
        -h|--help)
            echo "Usage: $0 [--out DIR | --continue-batch DIR] [--reuse-trained OLD_BATCH] [--dry-run | --validate] [config.yaml ...]"
            echo "Defaults: GlassNet coarse and oxide sintering, A2C, DQN-MC and DQN-bootstrap, seeds 11/29/47, popart and none."
            echo "Accepts YAML paths or names such as glassnet_coarse_2pct.yaml (searched in configs/)."
            echo "All jobs are submitted together. MAX_JOBS sets parallel workers (default 8)."
            echo "--dry-run prints commands. --validate checks them through the installed CLI."
            echo "--out writes runs and logs directly under DIR. Default: a fresh batch under ~/rl-matdesign-results/."
            echo "--continue-batch submits missing runs and preserves previous attempts."
            echo "Each run targets 1000 distinct candidates and preserves scenario exploration settings."
            echo "--reuse-trained loads final models when available and trains missing models."
            echo "For the oxide random forest, SINTER_MODEL can specify the absolute path to the model."
            echo "Set RL_MATDESIGN_REPO to use another checkout."
            echo "Activate your RL environment, or set RL_MATDESIGN_CLI to its rl-matdesign executable."
            exit 0 ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
    shift
done

caller_dir=$PWD
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
worker="$script_dir/${BASH_SOURCE[0]##*/}"
local_checkout="$HOME/Dropbox/Macbook_air/aisi/shanghai/reinforcement_learning/rl-matdesign/.worktrees/step1-normalization"
if [[ -n ${RL_MATDESIGN_REPO:-} ]]; then
    repo=$RL_MATDESIGN_REPO
elif [[ -f "$script_dir/../src/rl_matdesign/cli.py" ]]; then
    repo="$script_dir/.."
elif [[ -d "$local_checkout" ]]; then
    repo=$local_checkout
else
    repo=$PWD
fi
repo=$(cd -- "$repo" && pwd -P)
cli=${RL_MATDESIGN_CLI:-rl-matdesign}
command -v "$cli" >/dev/null 2>&1 || {
    echo "Cannot find $cli. Activate your RL environment and run: python -m pip install -e ." >&2
    exit 2
}
# The installed entry point may belong to another editable checkout. Select this
# checkout for the CLI and its child workers without changing the installation.
if [[ -f "$repo/src/rl_matdesign/cli.py" ]]; then
    export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
fi
configs=("$@")
if [[ $# -eq 0 ]]; then
    configs=(glassnet_coarse_2pct.yaml oxides_sinter.yaml)
fi
resolved_configs=()
work_dirs=()
for config in "${configs[@]}"; do
    if [[ -f "$config" && -r "$config" ]]; then
        if [[ "$config" != /* ]]; then config="$caller_dir/$config"; fi
        work_dirs+=("${RL_MATDESIGN_REPO:+$repo}")
    elif [[ "$config" != */* && -f "$caller_dir/configs/$config" ]]; then
        config="$caller_dir/configs/$config"
        work_dirs+=("$caller_dir")
    elif [[ "$config" != /* && -f "$repo/$config" ]]; then
        config="$repo/$config"
        work_dirs+=("$repo")
    elif [[ "$config" != */* && -f "$repo/configs/$config" ]]; then
        config="$repo/configs/$config"
        work_dirs+=("$repo")
    else
        echo "Cannot read YAML: $config" >&2
        echo "Looked in $caller_dir and $repo (including configs/). You can also pass an absolute YAML path." >&2
        exit 2
    fi
    resolved_configs+=("$config")
done
configs=("${resolved_configs[@]}")

parallel=${MAX_JOBS:-8}
[[ $parallel =~ ^[1-9][0-9]*$ ]] || { echo "MAX_JOBS must be a positive integer." >&2; exit 2; }
if [[ -n "$output_dir" && -n "$existing_batch" ]]; then
    echo "Choose either --out or --continue-batch." >&2
    exit 2
fi
if [[ -n "$output_dir" && "$output_dir" != /* ]]; then output_dir="$caller_dir/$output_dir"; fi
batch=${existing_batch:-${output_dir:-"$HOME/rl-matdesign-results/step1_preview"}}
if [[ -d "$batch" && -n "$output_dir" ]]; then existing_batch=$batch; fi
if [[ -n "$trained_batch" && -d "$batch" && "$trained_batch" -ef "$batch" ]]; then
    echo "Use a separate --out directory for the 1000-candidate comparison. --reuse-trained reads the original runs." >&2
    exit 2
fi
if [[ "$action" == submit ]]; then
    if [[ -z "$output_dir" && -z "$existing_batch" ]]; then
        mkdir -p "$HOME/rl-matdesign-results"
        batch=$(mktemp -d "$HOME/rl-matdesign-results/step1_$(date +%Y%m%d_%H%M%S)_XXXXXX")
    else
        mkdir -p -- "$batch"
    fi
    submission=$(mktemp -d "$batch/submission_XXXXXX")
    cp "$worker" "$submission/worker.sh"
    worker="$submission/worker.sh"
    queue="$submission/queue"
    : > "$queue"
fi

queued=0
skipped=0
for seed in "${seeds[@]}"; do
    for mode in "${modes[@]}"; do
        for index in "${!configs[@]}"; do
            config=${configs[index]}
            run_dir=${work_dirs[index]:-$caller_dir}
            name=${config##*/}
            scenario=${name%.*}
            for method in "${methods[@]}"; do
                case "$method" in
                    a2c) method_args=(--method a2c) ;;
                    dqn-mc) method_args=(--method dqn --dqn-target-mode mc) ;;
                    dqn-bootstrap) method_args=(--method dqn --dqn-target-mode bootstrap) ;;
                esac
                out="$batch/$scenario/$mode/$method/seed_$seed"
                args=("${method_args[@]}" --value-normalization "$mode"
                     --train-seed "$seed" --gen-seed "$seed")
                if [[ -n "$trained_batch" ]]; then
                    prior="$trained_batch/$scenario/$mode/$method/seed_$seed"
                    weights=qnet.pt
                    load_flag=--load-qnet
                    if [[ $method == a2c ]]; then weights=policy.pt; load_flag=--load-policy; fi
                    if [[ -s "$prior/$weights" ]]; then
                        [[ -s "$prior/std_scaler.bin" ]] || { echo "Missing scaler in $prior" >&2; exit 2; }
                        args+=(--only-generate "$load_flag" "$prior/$weights" --load-scaler "$prior/std_scaler.bin")
                        if [[ $mode == popart ]]; then
                            [[ -s "$prior/reward_calibration.json" ]] || { echo "Missing calibration in $prior" >&2; exit 2; }
                            args+=(--load-reward-calibration "$prior/reward_calibration.json")
                        fi
                    fi
                fi
                cmd=(bash "$worker" --run-one "$config" "$method" "$out" "${args[@]}")
                if [[ "$action" == validate ]]; then
                    (cd -- "$run_dir" && bash "$worker" --check-one "$config" "$method" "$out" "${args[@]}")
                    continue
                fi
                if [[ -n "$existing_batch" ]]; then
                    # The old launcher used a directory without a mode component.
                    old="$batch/$scenario/$method/seed_$seed/run_config.json"
                    if [[ -d "$out" ]] || { [[ -s "$old" ]] && python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    mode = json.load(f).get("value_normalization", "none")
sys.exit(mode != sys.argv[2])' "$old" "$mode"; }; then
                        echo "Previously started: $scenario $mode $method seed=$seed" >&2
                        skipped=$((skipped + 1))
                        continue
                    fi
                fi
                if [[ "$action" == preview ]]; then
                    printf 'cd -- %q && ' "$run_dir"
                    printf '%q ' "${cmd[@]}"
                    printf '\n'
                    continue
                fi
                queued=$((queued + 1))
                job="$submission/job_$queued.sh"
                printf 'cd -- %q || exit 1\nexec ' "$run_dir" > "$job"
                printf '%q ' "${cmd[@]}" >> "$job"
                printf '\n' >> "$job"
                printf '%s\0' "$out" "$job" >> "$queue"
            done
        done
    done
done

[[ "$action" == submit ]] || exit 0
echo "Batch: $batch"
if (( queued == 0 )); then
    echo "No new jobs. Previously started: $skipped."
    exit 0
fi
# xargs reads the whole queue independently of the calling terminal. Each worker
# claims its own output directory, including when two submissions overlap.
nohup xargs -0 -n 2 -P "$parallel" bash -c '
    out=$1
    job=$2
    mkdir -p -- "$(dirname -- "$out")" || exit 1
    if ! mkdir -- "$out" 2>/dev/null; then
        [[ -d "$out" ]] && exit 0
        echo "Could not create $out" >&2
        exit 1
    fi
    bash "$job" </dev/null > "$out/console.log" 2>&1
    result=$?
    printf "%s\n" "$result" > "$out/exit_code"
    printf "exit=%s %s\n" "$result" "$out"
    # In particular, never propagate 255: xargs treats it as "stop the queue".
    [[ $result == 0 ]]
' step1-worker < "$queue" > "$submission/dispatch.log" 2>&1 &
dispatcher=$!
echo "Submitted $queued jobs, skipped $skipped. Up to $parallel run concurrently (dispatcher PID $dispatcher)."
echo "Job results: $submission/dispatch.log"
