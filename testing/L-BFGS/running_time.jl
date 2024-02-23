include("OracleFunction.jl")
include("LBFGS.jl")
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

function helper(x::Union{Nothing, AbstractArray}, X, y)
    if isnothing(x)
        return (nothing, ones(eltype(y), size(X, 2)), nothing)
    end

    val = norm(X * x - y)
    return (val, X' * (X * x - y), nothing)
end

# warm up
LimitedMemoryBFGS(LeastSquaresF([1., 1], [1. 1; 0 1], [2., -1]))
LimitedMemoryBFGS(OracleF([1., 1],
        (x) -> norm([1. 1; 0 1] * x - [2., -1]),
        (x) -> inv(norm([1. 1; 0 -1] * x - [2., -1])) * [1. 1; 0 1]' * ([1. 1; 0 1] * x - [2., -1])
        ))

for i ∈ 1:rows
    e, en = [], []
    m = i * 200

    for j ∈ 1:runs
        A = rand(m, 300)
        [A[i, i] += 10 for i in minimum(size(A))]
        b = rand(m)

        push!(e, @elapsed begin
                sol = LimitedMemoryBFGS(LeastSquaresF(ones(size(A, 2)), A, b))
            end
        )

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

plt = plot(num_rows, run_time, seriestype = :scatter, label = "LBFGS", yscale = :log2)
plot!(plt, num_rows, run_time_non_quadratic, seriestype = :scatter, label = "LBFGS non quadratic")
xticks!(plt, num_rows .|> Int, map((x) -> (x[1]%2==1) ? x[2] : "", enumerate(num_rows .|> string)))
#ylims!(plt, (0, Inf))
mkpath("./results/")
savefig("./results/lbfgs_scalability.png")
