from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="lixiaowen/diffuEraser",
    local_dir="./weights/diffuEraser"
)

local_dir = snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    allow_patterns=["feature_extractor/*", "tokenizer/*", "*.json", "safety_checker/*", "scheduler/*", "text_encoder/*"],
    local_dir="./weights/stable-diffusion-v1-5"
)

local_dir = snapshot_download(
    repo_id="wangfuyun/PCM_Weights",
    local_dir="./weights/PCM_Weights" 
)

local_dir = snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    local_dir="./weights/sd-vae-ft-mse" 
)