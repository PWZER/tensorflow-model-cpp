## Replace workspace_name and dir_path as per your setup.
load("@com_grail_bazel_compdb//:aspects.bzl", "compilation_database")

compilation_database(
    name = "model_test_compdb",
    targets = [
        "//model_test:model_test",
    ],
    # exec_root = "$(bazel info bazel-bin)",
)

cc_library(
    name = "model_test_lib",
    srcs = [
        "main.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_binary(
    name = "model_test",
    stamp = 1,
    deps = [
        ":model_test_lib"
    ],
)
