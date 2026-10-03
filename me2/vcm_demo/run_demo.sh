#!/usr/bin/env bash
# Start the demo console from its preset, so the setup on the day is one word
# rather than a line of paths.
#
#   ./run_demo.sh fallback_hf   two seeds of vcm_hf, trained from random init on
#                               the class's Hugging Face dataset alone
#
# Anything after the preset is passed to demo.py, and a repeated flag overrides
# the preset's value:
#
#   ./run_demo.sh fallback_hf --none-bias 0.5          testers follow the example strip
#   ./run_demo.sh fallback_hf --wav recordings/*.wav   rehearse without a microphone
#   ./run_demo.sh fallback_hf --bench-log <id>         write the class benchmark's log
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
COMMON=(--ui --threshold 0.6)

case "${1:-}" in
  fallback_hf)
    # Two seeds of the same model, their probabilities averaged. Bias +1.1 is
    # chosen on the validation speakers - the most (correct - wrong) that
    # still leaves 90% of out-of-scope requests alone
    MODELS=(deploy/vcm_hf deploy/vcm_hf_s2)
    BIAS=1.1
    ;;
  *)
    echo "usage: $0 fallback_hf [extra demo.py arguments]" >&2
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
