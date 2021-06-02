load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


def model_test_workspace():
    # https://github.com/grailbio/bazel-compilation-database
    http_archive(
        name = "com_grail_bazel_compdb",
        strip_prefix = "bazel-compilation-database-master",
        urls = ["https://github.com/grailbio/bazel-compilation-database/archive/master.tar.gz"],
        sha256 = "df63430df795293fe49d2b71a731278038da1d95f00e3c2b9efd141126c1f3d9",
    )

    git_repository(
        name = "org_tensorflow",
        remote = "https://github.com/tensorflow/tensorflow.git",
        commit = "a4dfb8d1a71385bd6d122e4f27f86dcebb96712d",  # v2.5.0
        recursive_init_submodules = True,
    )
