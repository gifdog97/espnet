#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

ORIGINAL_ARGS=("$@")

fs=22050
n_fft=1024
n_shift=256

opts=
if [ "${fs}" -eq 22050 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"
datadir="data_orig"

# 引数の解析
while [[ $# -gt 0 ]]; do
  case "$1" in
    --datadir)
      datadir="$2"
      shift 2
      ;;
    *)
      shift 1
      ;;
  esac
done

set -- "${ORIGINAL_ARGS[@]}"

train_config=conf/train.yaml
inference_config=conf/decode.yaml

# g2p=g2p_en # Include word separator
# g2p=g2p_en_no_space # Include no word separator
g2p=none # Just separate by space
cleaner=none

./tts.sh \
    --lang en \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --token_type phn \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "${datadir}/${train_set}/text" \
    ${opts} "$@"
