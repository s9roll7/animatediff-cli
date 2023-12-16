#!/bin/bash

mkdir -p data/share/Stable-diffusion/

pget https://civitai.com/api/download/models/78775 data/share/Stable-diffusion/toonyou_beta3.safetensors || true
pget https://civitai.com/api/download/models/72396 data/share/Stable-diffusion/lyriel_v16.safetensors || true
pget https://civitai.com/api/download/models/71009 data/share/Stable-diffusion/rcnzCartoon3d_v10.safetensors || true
pget https://civitai.com/api/download/models/79068 data/share/Stable-diffusion/majicmixRealistic_v5Preview.safetensors || true
pget https://civitai.com/api/download/models/29460 data/share/Stable-diffusion/realisticVisionV40_v20Novae.safetensors || true

# Download Motion_Module models
wget -O data/models/motion-module/mm_sd_v14.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt || true
wget -O data/models/motion-module/mm_sd_v15.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt || true
wget -O data/models/motion-module/mm_sd_v15_v2.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt || true
