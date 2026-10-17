#!/usr/bin/env bash
# Full, reproducible GEMM benchmark run for docs/performance.md, the README, and plots.
#
# Checks that the machine is in a controlled state (performance governor, turbo off, SMT off, idle),
# records the environment, and runs every benchmark group in its own process with JSON output:
#
#   ladder_f64 / ladder_f32       naive ijk -> ikj -> blocked -> AVX2 -> AVX2 + threads, 64..4096
#   openblas_f64 / openblas_f32   OpenBLAS 1 thread and N threads (separate process: its worker
#                                 threads spin after each call and would steal cores from ours)
#   threads                       thread scaling 1..6 (and 12 with SMT) plus rectangular shapes
#   loop_orders                   the six scalar loop orders (Experiment 4 in docs/performance.md)
#   perf/                         perf stat counters (GHz, IPC, cache misses, stalls) per variant
#
# Usage: tools/run_benchmarks.sh [--quick] [--cpus LIST] [--reps N] [--force] [--no-build] [--out DIR]
#   --quick      3 repetitions, naive ijk only up to 1024, no 2048 loop orders (~10 min)
#   --cpus LIST  run under `taskset -c LIST`, e.g. 0-5 (use isolated cores if you have them)
#   --reps N     repetitions per benchmark (default 5)
#   --force      run even if the environment checks fail (results are then marked uncontrolled)
#   --no-build   use the existing build/release binaries
#   --out DIR    output directory (default benchmark-results/<date>-<commit>)
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

quick=0 force=0 build=1 reps=5 cpus="" out=""
while (($#)); do
    case $1 in
        --quick) quick=1 reps=3 ;;
        --force) force=1 ;;
        --no-build) build=0 ;;
        --cpus) cpus=$2; shift ;;
        --reps) reps=$2; shift ;;
        --out) out=$2; shift ;;
        -h | --help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
[[ -n $(git status --porcelain 2>/dev/null) ]] && commit="${commit}-dirty"
out=${out:-benchmark-results/$(date +%Y%m%d-%H%M%S)-$commit}
bin=build/release/benchmarks/linalg_benchmarks

# ---------------------------------------------------------------- environment checks
read_or() { cat "$1" 2>/dev/null || echo "$2"; }

failures=()
check() { # name, ok (0/1), detail, fix
    if [[ $2 == 1 ]]; then
        printf '  [ok]   %-22s %s\n' "$1" "$3"
    else
        printf '  [FAIL] %-22s %s\n         fix: %s\n' "$1" "$3" "$4"
        failures+=("$1")
    fi
}

echo "Environment checks:"
# Offline CPUs (for example the SMT siblings after `smt/control=off`) keep their cpufreq files but
# reading them fails with EBUSY; only online CPUs matter, so read errors are ignored.
governors=$( (cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true) |
    sort -u | tr '\n' ' ')
check governor "$([[ $governors == "performance " ]] && echo 1 || echo 0)" "$governors" \
    "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"

if [[ -e /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    no_turbo=$(read_or /sys/devices/system/cpu/intel_pstate/no_turbo "?")
    check "turbo off" "$([[ $no_turbo == 1 ]] && echo 1 || echo 0)" "intel_pstate/no_turbo=$no_turbo" \
        "echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo"
else
    boost=$(read_or /sys/devices/system/cpu/cpufreq/boost "?")
    check "turbo off" "$([[ $boost == 0 ]] && echo 1 || echo 0)" "cpufreq/boost=$boost" \
        "echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost"
fi

smt=$(read_or /sys/devices/system/cpu/smt/active "?")
check "SMT off" "$([[ $smt == 0 ]] && echo 1 || echo 0)" "smt/active=$smt" \
    "echo off | sudo tee /sys/devices/system/cpu/smt/control"

# isolcpus removes CPUs from scheduler load balancing: threads are not migrated onto them, so a
# 6-thread run on a machine with isolated cores silently shares fewer cores (and with SMT off,
# isolcpus=2,4 leaves only 4 of 6 cores to the benchmark).
isolated=$(read_or /sys/devices/system/cpu/isolated "")
check "no isolated CPUs" "$([[ -z $isolated ]] && echo 1 || echo 0)" "isolated=${isolated:-none}" \
    "boot once without isolcpus/nohz_full/rcu_nocbs (GRUB: press e, remove them, Ctrl+X)"

# Busy share of all online CPUs over a few seconds, from /proc/stat. The load average is a poor
# idleness test on a desktop: it is smoothed over a minute and counts short spikes (including the
# shell that starts this script), so it can read 1.5 while the CPUs are 96% idle.
cpu_busy_percent() {
    local _ user nice system idle iowait irq softirq steal total1 idle1 total2 idle2
    read -r _ user nice system idle iowait irq softirq steal _ </proc/stat
    total1=$((user + nice + system + idle + iowait + irq + softirq + steal)) idle1=$((idle + iowait))
    sleep 3
    read -r _ user nice system idle iowait irq softirq steal _ </proc/stat
    total2=$((user + nice + system + idle + iowait + irq + softirq + steal)) idle2=$((idle + iowait))
    echo $((100 * ((total2 - total1) - (idle2 - idle1)) / (total2 - total1)))
}
busy=$(cpu_busy_percent)
check "idle machine" "$((busy < 10 ? 1 : 0))" "CPU busy ${busy}% over 3 s (load average $(cut -d' ' -f1 /proc/loadavg))" \
    "close IDEs, browsers, containers (docker ps); top -o %CPU shows what is running"

paranoid=$(read_or /proc/sys/kernel/perf_event_paranoid 4)
perf_ok=0
command -v perf >/dev/null && ((paranoid <= 1)) && perf_ok=1
check "perf counters" "$perf_ok" "perf_event_paranoid=$paranoid" \
    "sudo sysctl kernel.perf_event_paranoid=1   (only the perf/ section needs it)"

uncontrolled=0
if ((${#failures[@]})); then
    if ((force)); then
        echo "Continuing with --force: results are marked UNCONTROLLED."
        uncontrolled=1
    else
        echo "Environment is not controlled (${failures[*]}). Fix the items above or pass --force." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------- build and record environment
if ((build)); then
    cmake --preset release >/dev/null
    cmake --build --preset release --parallel "$(nproc)" >/dev/null
fi
[[ -x $bin ]] || { echo "missing $bin; build the release preset first" >&2; exit 1; }

mkdir -p "$out/perf"
runner=()
[[ -n $cpus ]] && runner=(taskset -c "$cpus")

{
    echo "date:          $(date -Iseconds)"
    echo "commit:        $commit"
    echo "uncontrolled:  $uncontrolled (failed checks: ${failures[*]:-none})"
    echo "options:       quick=$quick reps=$reps cpus=${cpus:-all}"
    echo "governor:      $governors"
    echo "no_turbo:      $(read_or /sys/devices/system/cpu/intel_pstate/no_turbo n/a)"
    echo "smt active:    $smt"
    echo "online cpus:   $(read_or /sys/devices/system/cpu/online "?")"
    echo "isolated cpus: ${isolated:-none}"
    echo "cpu busy:      ${busy}% over 3 s before the run"
    echo "load average:  $(cat /proc/loadavg)"
    echo "kernel:        $(uname -srmv)"
    echo "cmdline:       $(cat /proc/cmdline)"
    echo "compiler:      $(grep -m1 '^CMAKE_CXX_COMPILER:' build/release/CMakeCache.txt | cut -d= -f2) $(
        "$(grep -m1 '^CMAKE_CXX_COMPILER:' build/release/CMakeCache.txt | cut -d= -f2)" --version | head -1)"
    echo "build type:    $(grep -m1 '^CMAKE_BUILD_TYPE:' build/release/CMakeCache.txt | cut -d= -f2)"
    echo "openblas:      $(pkg-config --modversion openblas 2>/dev/null || echo 'not found')"
    echo
    lscpu
    echo
    echo "top processes:"
    ps -eo pcpu,comm --sort=-pcpu | head -8 || true # head may close the pipe early (SIGPIPE)
} >"$out/env.txt"

# ---------------------------------------------------------------- benchmark groups
common=(--benchmark_repetitions="$reps" --benchmark_enable_random_interleaving=true
        --benchmark_display_aggregates_only=true --benchmark_min_time=0.5s --benchmark_out_format=json)

run_group() { # name, filter
    echo
    echo "== $1"
    # Context (stderr) goes to the log as well; the console shows medians, or every row if reps=1.
    local shown='_median|ERROR|SKIP'
    ((reps == 1)) && shown='^BM_|ERROR|SKIP'
    "${runner[@]}" "$bin" --benchmark_filter="$2" --benchmark_out="$out/$1.json" "${common[@]}" 2>&1 |
        tee "$out/$1.txt" | grep -E "$shown" || true
}

ladder='ScalarIkj|BlockedPortable|BlockedAvx2|BlockedAvx2Mt'
for type in f64 f32; do
    if ((quick)); then
        run_group "ladder_$type" "^BM_Gemm($ladder)_$type/|^BM_GemmNaiveIjk_$type/([0-9]{2,3}|1024)/"
    else
        run_group "ladder_$type" "^BM_Gemm(NaiveIjk|$ladder)_$type/"
    fi
done

# Listing into a variable first: `binary | grep -q` can kill the binary with SIGPIPE, and
# pipefail would then report the pipeline as failed.
registered=$("$bin" --benchmark_list_tests)
if grep -q '^BM_GemmOpenBlas' <<<"$registered"; then
    run_group openblas_f64 '^BM_GemmOpenBlas(Mt)?_f64/'
    run_group openblas_f32 '^BM_GemmOpenBlas(Mt)?_f32/'
else
    echo "OpenBLAS baselines not built (install libopenblas-dev and reconfigure)"
fi

run_group threads '^BM_MatMulBlockedAvx2FmaThreads/'
if ((quick)); then
    run_group loop_orders '^BM_MatMul(Ijk|Ikj|Jik|Jki|Kij|Kji)Kernel/[0-9]+/[0-9]+/[0-9]+$'
else
    run_group loop_orders '^BM_MatMul(Ijk|Ikj|Jik|Jki|Kij|Kji)Kernel/'
fi

# ---------------------------------------------------------------- perf counters per variant
# One size for all variants (naive ijk at 2048 would take minutes), double precision.
if ((perf_ok)); then
    events=task-clock,cycles,instructions,L1-dcache-load-misses,LLC-loads,LLC-load-misses
    events+=,cycle_activity.stalls_l2_miss,context-switches,page-faults
    for variant in NaiveIjk ScalarIkj BlockedPortable BlockedAvx2 BlockedAvx2Mt OpenBlas OpenBlasMt; do
        filter="^BM_Gemm${variant}_f64/1024/"
        grep -qE "$filter" <<<"$registered" || continue
        echo "== perf $variant"
        "${runner[@]}" perf stat -x, -e "$events" -o "$out/perf/$variant.csv" \
            "$bin" --benchmark_filter="$filter" --benchmark_min_time=3s \
            --benchmark_out="$out/perf/$variant.json" --benchmark_out_format=json >/dev/null
    done
fi

echo
echo "Results in $out"
echo "Restore the machine afterwards, for example:"
echo "  echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
echo "  echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo"
echo "  echo on | sudo tee /sys/devices/system/cpu/smt/control"
