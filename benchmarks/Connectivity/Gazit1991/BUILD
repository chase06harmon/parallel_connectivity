licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Connectivity",
    hdrs = ["Connectivity.h", "Connectivity_Tests.h"],
    deps = [
        "//benchmarks/Connectivity:common",
        "//benchmarks/Connectivity/WorkEfficientSDB14:Connectivity",
        "//gbbs",
    ],
)

cc_binary(
    name = "Connectivity_main",
    srcs = ["Connectivity.cc"],
    deps = [":Connectivity"],
)

cc_binary(
    name = "Comparison_main",
    srcs = ["Comparison.cc"],
    deps = [":Connectivity"],
)

cc_binary(
    name = "Sparse2Dense",
    srcs = ["Sparse2Dense.cc"],
    deps = [":Connectivity"],
)

cc_binary(
    name = "TestRunner_main",
    srcs = ["TestRunner.cc"],
    deps = [":Connectivity"],
)