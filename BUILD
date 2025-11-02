load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
        "//benchmarks/Connectivity/WorkEfficientSDB14:Connectivity_main": "--macos_minimum_os=11.0 --cxxopt=-UPARLAY_USE_STD_ALLOC",
    },
    exclude_headers = "external",
    exclude_external_sources = True,
)
