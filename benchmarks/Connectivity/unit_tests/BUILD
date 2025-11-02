licenses(["notice"])

load("//internal_tools:build_defs.bzl", "gbbs_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

gbbs_cc_test(
    name = "test_cc",
    srcs = ["test_cc.cc"],
    data = ["//inputs:graphs"],
    size = "small",
    deps = [
        "//benchmarks/Connectivity/Gazit1991:Connectivity",
        "//benchmarks/Connectivity/WorkEfficientSDB14:Connectivity",
        "//gbbs:graph_io",
        "//gbbs:graph",
        "//gbbs:macros",
        "//gbbs/unit_tests:graph_test_utils",
        "@googletest//:gtest_main",
    ],
)
