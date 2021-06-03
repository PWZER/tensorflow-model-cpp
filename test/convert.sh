#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL="1"

# ================================ model v1 ========================================================
# saved_model_cli show --all --dir savedmodel-v1/

# model v1 to frozen pb
python3 -m tensorflow.python.tools.freeze_graph \
    --input_saved_model_dir=savedmodel-v1/ \
    --output_node_names="embedding_lookup/Identity_1" \
    --output_graph=savedmodel-v1/frozen_model.pb

# model v1 to onnx
python3 -m tf2onnx.convert  --saved-model savedmodel-v1 --output savedmodel-v1/model.onnx --opset 11

# ================================ model v2 ========================================================
# saved_model_cli show --all --dir savedmodel-v2/

# model v2 to frozen pb
python3 -m tensorflow.python.tools.freeze_graph \
    --input_saved_model_dir=savedmodel-v2/ \
    --output_node_names="StatefulPartitionedCall" \
    --output_graph=savedmodel-v2/frozen_model.pb

# model v2 to onnx
python3 -m tf2onnx.convert  --saved-model savedmodel-v2 --output savedmodel-v2/model.onnx --opset 11
