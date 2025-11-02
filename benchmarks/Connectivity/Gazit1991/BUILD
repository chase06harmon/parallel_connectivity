licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Connectivity",
    hdrs = ["Connectivity.h"],
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
