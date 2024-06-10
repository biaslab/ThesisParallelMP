# ThesisParallelMP
This repository contains the code and results of the MSc graduation project of Mattia Vicari about parallel computing in message passing-based Bayesian inference programs

## Setup
In order to run successfully the different benchmarks the only requirements is to have Jupiter kernels with Julia 1.10.0.

In order to create a Jupiter kernel with multiple threads for the multi-threading benchmarks the following commands should be used in the Julia REPL.

First we need to ensure that `IJulia` is installed and used.
```Julia
] add IJulia
```

```Julia
using IJulia
```

Finally the kernel can be created with the desired number of threads (6 in the example).
```Julia
installkernel("Julia 6 Threads", env=Dic("JULIA_NUM_THREADS"=>"6"))
```

## Benchmarks run

Once the desired Jupiter kernel is created all the benchmark notebooks in the `benchmarks-dirstibuted` and `benchmarks-multithreading` folders can be executed.

## High-quality pictures

In the `hq-pictures` folder high-quality pictures of the report's plots can be found for better visualization.

## Code

The reusable code that enables parallelism in `RxInfer` can be found in the files `distributed.jl` and `multithreading.jl`.
