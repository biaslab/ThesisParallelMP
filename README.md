# ThesisParallelMP

This repository contains the code and results from Mattia Vicari’s MSc graduation project on parallel computing in message passing-based Bayesian inference programs.

## Setup

To successfully run the benchmarks, the only requirement is to have Jupyter kernels with Julia 1.10. We recommend installing Julia using [juliaup](https://github.com/JuliaLang/juliaup), which is compatible with all major operating systems. Once Julia is installed, you need to install the IJulia package to create the necessary Jupyter kernels. Run the following commands in the Julia REPL:

```Julia
] add IJulia
```

For creating a Jupyter kernel with multiple threads for the multithreading benchmarks, use these commands in the Julia REPL:

```julia
julia> using IJulia

julia> installkernel("Julia 6 Threads", env=Dic("JULIA_NUM_THREADS"=>"6"))
```

## Setting up the environment

Before running the experiments, it’s crucial to set up the environment. Execute the following commands in the Julia REPL:

```julia
] activate .
] instantiate
```

This ensures that all required packages are installed and the environment is ready for running the benchmarks.

## Running benchmarks

Once the desired Jupyter kernel is created, execute all benchmark notebooks located in the `benchmarks-distributed` and `benchmarks-multithreading` folders. You can also do this from the Julia REPL with:

```julia
julia> using IJulia

julia> notebook(dir = pwd()) # jupyterlab(dir = pwd())
```

## High-quality pictures

The `hq-pictures` folder contains high-quality images of the plots from the report for better visualization.

## Code

The reusable code that enables parallelism in `RxInfer` can be found in the files `distributed.jl` and `multithreading.jl`.
