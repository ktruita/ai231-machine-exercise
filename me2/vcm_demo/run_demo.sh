#!/usr/bin/env bash
# Start the demo console from its preset, so the setup on the day is one word
# rather than a line of paths.
#
#   ./run_demo.sh fallback    two seeds of vcm_mined_fsc, trained from random init
#
# Anything after the preset is passed to demo.py, and a repeated flag overrides
# the preset's value:
#
#   ./run_demo.sh fallback --none-bias -0.5          testers follow the example strip
#   ./run_demo.sh fallback --wav recordings/*.wav    rehearse without a microphone
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
COMMON=(--ui --threshold 0.6)

case "${1:-}" in
  fallback)
    # Two seeds of the same model: 21% fewer wrong actions than one, with no
    # loss on unseen speakers. Bias -0.2 is chosen on validation - the most
    # (correct - wrong) that still leaves 90% of out-of-scope requests alone
    MODELS=(deploy/vcm_mined_fsc deploy/vcm_mined_fsc_s2)
    BIAS=-0.2
    ;;
  *)
    echo "usage: $0 fallback [extra demo.py arguments]" >&2
    exit 2
    ;;
esac
PRESET="$1"
shift

for model in "${MODELS[@]}"; do
  if [[ ! -f "$model/model.onnx" ]]; then
    echo "preset '$PRESET': $model/model.onnx not found" >&2
    exit 1
  fi
done

echo "preset   $PRESET: ${MODELS[*]}  (none bias $BIAS)"
exec "$PYTHON" demo.py "${COMMON[@]}" --model "${MODELS[@]}" --none-bias "$BIAS" "$@"
