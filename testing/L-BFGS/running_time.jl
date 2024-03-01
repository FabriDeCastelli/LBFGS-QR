include("../../L-BFGS/OracleFunction.jl")
include("../../L-BFGS/LBFGS.jl")
using ..LBFGS
using ..OracleFunction
using Plots: plot, plot!, savefig, ylims!, xticks!
using LinearAlgebra: norm, I
using Statistics: mean

rows = 20
runs = 5

num_rows = []
run_time = []
run_time_non_quadratic = []

# warm up
LimitedMemoryBFGS(LeastSquaresF([1., 1], [1. 1; 0 1], [2., -1]))
LimitedMemoryBFGS(OracleF([1., 1],
        (x) -> norm([1. 1; 0 1] * x - [2., -1]),
        (x) -> inv(norm([1. 1; 0 -1] * x - [2., -1])) * [1. 1; 0 1]' * ([1. 1; 0 1] * x - [2., -1])
        ))

for i ∈ 1:rows
    e, en = [], []
    m = i * 100

    for j ∈ 1:runs
        A = rand(m, 300)
        [A[i, i] += 10 for i in minimum(size(A))]
        b = rand(m)

        acd = @elapsed begin
            sol = LimitedMemoryBFGS(LeastSquaresF(ones(size(A, 2)), A, b))
        end

        push!(e, acd)

        push!(en, @elapsed begin
            sol = LimitedMemoryBFGS(OracleF(
                        ones(size(A, 2)),
                        (x) -> norm(A * x - b),
                        (x) -> inv(norm(A * x - b)) * A' * (A * x - b)
                    ))
            end
        )
    end
    println("Time for $m rows: $(mean(e)) ($(mean(en)) non quadratic)")

    push!(num_rows, m)
    push!(run_time, mean(e))
    push!(run_time_non_quadratic, mean(en))
end
