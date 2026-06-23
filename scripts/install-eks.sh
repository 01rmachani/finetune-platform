#!/usr/bin/env bash
#
# One-command install of the Fine-Tuning Platform on an EKS (or any) cluster using
# the prebuilt PUBLIC image (ghcr.io/t4tarzan/finetune-platform:latest, amd64).
# No docker build, no pull secret.
#
# Prereqs: kubectl (pointed at your cluster), helm, a StorageClass (default gp3),
# an amd64 node, and egress to ghcr.io (+ huggingface.co for the base model).
#
# Usage:
#   bash scripts/install-eks.sh
#   NS=ml STORAGE_CLASS=gp3 BASE_MODEL=qwen2.5:1.5b bash scripts/install-eks.sh
set -euo pipefail

NS="${NS:-finetune}"
STORAGE_CLASS="${STORAGE_CLASS:-gp3}"
BASE_MODEL="${BASE_MODEL:-qwen2.5:0.5b}"
CHART="$(cd "$(dirname "$0")/.." && pwd)/charts/finetune-platform"

echo "==> Installing release 'finetune-platform' into namespace '$NS' (storageClass=$STORAGE_CLASS)"
helm upgrade --install finetune-platform "$CHART" \
  --namespace "$NS" --create-namespace \
  --set persistence.storageClass="$STORAGE_CLASS" \
  --set ollama.enabled=true

echo "==> Waiting for rollout (first boot downloads the base model — a few minutes)"
kubectl -n "$NS" rollout status deploy/finetune-platform --timeout=900s

echo "==> Pulling base model '$BASE_MODEL' into the ollama sidecar"
kubectl -n "$NS" exec -c ollama deploy/finetune-platform -- ollama pull "$BASE_MODEL" || \
  echo "   (pull failed — do it later: kubectl -n $NS exec -c ollama deploy/finetune-platform -- ollama pull $BASE_MODEL)"

cat <<EOF

==> Done. Open the UI:
    kubectl -n $NS port-forward svc/finetune-platform 7100:7100
    # then browse http://localhost:7100

    Or expose it:  helm upgrade finetune-platform "$CHART" -n $NS --reuse-values --set service.type=LoadBalancer
    Sub-path ingress (nginx): add --set ingress.enabled=true --set ingress.className=nginx --set basePath=/finetune-platform
EOF
