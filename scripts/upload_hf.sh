#!/usr/bin/env bash
# ==============================================================================
# Script to upload 3024-d Korean Utterance Embedding model to Hugging Face Hub
# ==============================================================================

set -euo pipefail

MODEL_DIR="${1:-models}"
HF_REPO_ID="${2:-none-labs/korean-embedding-3024d}"

echo "================================================================================"
echo "Uploading 3024-d Korean Embedding Model to Hugging Face: ${HF_REPO_ID}"
echo "================================================================================"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Warning: HF_TOKEN environment variable is not set."
    echo "Please set HF_TOKEN to authenticate with Hugging Face Hub if pushing."
fi

mkdir -p "${MODEL_DIR}"

# Generate model configuration file
cat <<EOF > "${MODEL_DIR}/model_config.json"
{
  "model_type": "korean-utterance-embedding",
  "architecture": "HierarchicalMultiAspectEmbedding",
  "embedding_dimension": 3024,
  "subspaces": {
    "literal_semantics": {
      "dimension": 1536,
      "offset": 0,
      "description": "Morphological & syntactic literal proposition"
    },
    "speaker_intent": {
      "dimension": 512,
      "offset": 1536,
      "classes": 8,
      "description": "8-class communicative intent & speech act"
    },
    "social_stance": {
      "dimension": 512,
      "offset": 2048,
      "description": "Formal vs informal relationship & age hierarchy"
    },
    "pragmatic_nuance": {
      "dimension": 464,
      "offset": 2560,
      "description": "Irony, sarcasm, and attitude polarity"
    }
  },
  "mrl_dimensions": [512, 1024, 1536, 3024],
  "l2_normalized": true,
  "runtime": "pure-golang",
  "version": "1.0.0"
}
EOF

echo "==> Generated model configuration: ${MODEL_DIR}/model_config.json"

if command -v huggingface-cli &> /dev/null; then
    echo "==> Uploading artifacts via huggingface-cli..."
    huggingface-cli upload "${HF_REPO_ID}" "${MODEL_DIR}" .
    echo "==> Upload completed successfully!"
else
    echo "==> huggingface-cli not installed. Prepared model artifacts are in: ${MODEL_DIR}"
    echo "    To install: pip install huggingface_hub"
fi
