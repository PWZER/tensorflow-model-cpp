#!/bin/bash

# Step 1 build model_test
bazel build //model_test:model_test

# Step 2 build compile_commands.json
bazel build //model_test:model_test_compdb
execroot=$(bazel info execution_root)
outfile="$(bazel info bazel-bin)/model_test/compile_commands.json"
sed -i.bak "s@__EXEC_ROOT__@${execroot}@" "${outfile}"
echo "Compilation Database: ${outfile}"
