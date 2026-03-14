#!/bin/bash
# Setup script for running TEM training on Google Cloud with A100 GPU
#
# Usage (from your local machine):
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. gcloud auth login
#   3. gcloud config set project YOUR_PROJECT_ID
#   4. bash setup_gcloud.sh
#
# This will create a VM, SSH in, and start training.
# To check on training later: gcloud compute ssh tem-train --zone=us-central1-a

set -e

INSTANCE_NAME="tem-train"
ZONE="us-central1-a"
MACHINE_TYPE="a2-highgpu-1g"  # 1x A100 40GB

echo "=== Creating VM with A100 GPU ==="
gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True

echo "=== Waiting for VM to be ready ==="
sleep 30

echo "=== Running setup and training on VM ==="
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="$(cat <<'REMOTE_SCRIPT'
set -e

echo "=== Installing dependencies ==="
pip install gymnasium[mujoco] mujoco 2>&1 | tail -3

echo "=== Cloning repo ==="
git clone https://github.com/YX234/tem-generalization.git
cd tem-generalization

echo "=== Verifying GPU ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo "=== Starting training in tmux ==="
tmux new-session -d -s train "cd ~/tem-generalization && python3 train.py 2>&1 | tee train_output.log"

echo ""
echo "=== Training started in tmux session 'train' ==="
echo "To monitor: gcloud compute ssh tem-train --zone=us-central1-a -- tmux attach -t train"
echo "To check logs: gcloud compute ssh tem-train --zone=us-central1-a -- tail -f ~/tem-generalization/train_output.log"
REMOTE_SCRIPT
)"

echo ""
echo "=== VM is running. Training has started. ==="
echo ""
echo "Useful commands:"
echo "  Monitor training:  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- tmux attach -t train"
echo "  Check logs:        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- tail -20 ~/tem-generalization/train_output.log"
echo "  Download results:  gcloud compute scp --recurse $INSTANCE_NAME:~/tem-generalization/runs ./runs --zone=$ZONE"
echo "  Stop VM (save \$):  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo "  Delete VM:         gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
