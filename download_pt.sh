MODEL_NAME="Vayan0/VFPD"

# hf download "Vayan0/VFPD" --repo-type model \
#     --include "OpenWAM/real/piper_banana_ft/*" \
#     --exclude "OpenWAM/real/piper_banana_ft/checkpoint_step_2000.safetensors" \
#     --local-dir '/media/ubun/16T/ming/openwam'

# hf download "Vayan0/VFPD" --repo-type model \
#     --include "OpenWAM/real/umt5_xxl/*" \
#     --local-dir '/media/ubun/16T/ming/openwam'
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT
hf download "Vayan0/VFPD" --repo-type model \
    --include "OpenWAM/real/piper_book_vstack_ft/*" \
    --exclude "OpenWAM/real/piper_book_vstack_ft/checkpoint_step_2000.safetensors" \
    --exclude "OpenWAM/real/piper_book_vstack_ft/checkpoint_step_4000.safetensors" \
    --local-dir '/media/ubun/16T/ming/openwam' 