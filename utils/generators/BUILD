licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "BarabasiAlbert",
    srcs = ["BarabasiAlbert.cc"],
    deps = [
        "BarabasiAlbert_library",
        "//gbbs",
    ],
)

cc_binary(
    name = "RMAT",
    srcs = ["RMAT.cc"],
    deps = [
        "RMAT_library",
        "//gbbs",
    ],
)

cc_binary(
    name = "RMAT_csr",
    srcs = ["RMAT_csr.cc"],
    deps = [
        "RMAT_library",
        "//gbbs",
    ],
)

cc_binary(
    name = "line",
    srcs = ["line.cc"],
    deps = [
        "line_library",
        "//gbbs",
    ],
)


cc_library(
    name = "BarabasiAlbert_library",
    hdrs = ["barabasi_albert.h"],
    deps = [
        "//gbbs",
        "//gbbs:macros",
    ],
)

cc_library(
    name = "RMAT_library",
    hdrs = ["rmat.h"],
    deps = [
        "//gbbs",
        "//gbbs:macros",
    ],
)

cc_binary(
    name = "RegularRandom",
    srcs = ["RegularRandom.cc"],
    deps = [
        "RegularRandom_library",
        "//gbbs",
    ],
)

cc_library(
    name = "RegularRandom_library",
    hdrs = ["regular_random.h"],
    deps = [
        "//gbbs",
        "//gbbs:macros",
    ],
)

cc_library(
    name = "line_library",
    hdrs = ["line.h"],
    deps = [
        "//gbbs",
        "//gbbs:macros",
    ],
)