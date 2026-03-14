#!/bin/bash
# Setup script for running TEM training on Google Cloud with GPU
#
# Tries A100 first across multiple zones, falls back to T4 if unavailable.
#
# Usage (from your local machine):
#   1. Install gcloud CLI
#   2. gcloud auth login
#   3. gcloud config set project YOUR_PROJECT_ID
#   4. bash setup_gcloud.sh

set -e

INSTANCE_NAME="tem-train"
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"

# GPU configs to try in order (type, machine-type, zone)
CONFIGS=(
  "nvidia-tesla-a100,a2-highgpu-1g,us-central1-a"
  "nvidia-tesla-a100,a2-highgpu-1g,us-east1-b"
  "nvidia-tesla-a100,a2-highgpu-1g,us-west1-b"
  "nvidia-tesla-a100,a2-highgpu-1g,europe-west4-a"
  "nvidia-tesla-t4,n1-standard-8,us-central1-a"
  "nvidia-tesla-t4,n1-standard-8,us-east1-b"
  "nvidia-tesla-t4,n1-standard-8,us-west1-b"
  "nvidia-tesla-t4,n1-standard-8,europe-west4-a"
  "nvidia-tesla-t4,n1-standard-8,us-central1-f"
)

CREATED=false
for config in "${CONFIGS[@]}"; do
  IFS=',' read -r GPU MACHINE ZONE <<< "$config"
  echo "=== Trying $GPU in $ZONE ($MACHINE) ==="
  if gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE \
    --accelerator=type=$GPU,count=1 \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata=install-nvidia-driver=True 2>&1; then
    CREATED=true
    echo "=== Success: $GPU in $ZONE ==="
    break
  else
    echo "--- $GPU unavailable in $ZONE, trying next... ---"
  fi
done

if [ "$CREATED" = false ]; then
  echo "ERROR: Could not create VM in any zone. Try again later."
  exit 1
fi

echo "=== Waiting for VM to be ready ==="
sleep 30

echo "=== Running setup and training on VM ==="
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="$(cat <<'REMOTE_SCRIPT'
set -e

echo "=== Installing dependencies ==="
pip install gymnasium[mujoco] mujoco 2>&1 | tail -3

echo "=== Cloning repo ==="
if [ ! -d ~/tem-generalization ]; then
  git clone https://github.com/YX234/tem-generalization.git
fi
cd tem-generalization

echo "=== Verifying GPU ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo "=== Starting training in tmux ==="
tmux new-session -d -s train "cd ~/tem-generalization && python3 train.py 2>&1 | tee train_output.log"

echo ""
echo "=== Training started in background tmux session ==="
REMOTE_SCRIPT
)"

echo ""
echo "============================================"
echo "  VM running: $INSTANCE_NAME ($GPU in $ZONE)"
echo "============================================"
echo ""
echo "Useful commands:"
echo "  Monitor training:  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- tmux attach -t train"
echo "  Check logs:        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- tail -20 ~/tem-generalization/train_output.log"
echo "  Download results:  gcloud compute scp --recurse $INSTANCE_NAME:~/tem-generalization/runs ./runs --zone=$ZONE"
echo "  Stop VM (save \$):  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo "  Delete VM:         gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
