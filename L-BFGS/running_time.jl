include("LBFGS.jl")
using .LBFGS
using Plots: plot, savefig
using LinearAlgebra: norm

runs = 10
num_rows = zeros(runs)
run_time = zeros(runs)

function helper(x::Union{Nothing, AbstractArray}, X, y)
    if isnothing(x)
        return (nothing, ones(eltype(y), size(X, 2)), nothing)
    end

    val = norm(X * x - y)
    return (val, inv(val) * (X' * (X * x - y)), nothing)
end

# warm up
LimitedMemoryBFGS(x -> helper(x, [0. 1; 1 0], [1.; 1]))

for i ∈ 0:runs
    m = 500 + i * 500
    A = rand(m, 300)
    b = rand(m)

    f = x -> let X = A, y = b
        helper(x, X, y)
    end
    
    elapsed_time = @elapsed begin
        sol = LimitedMemoryBFGS(f)
    end 
    println("Time for $m rows: $elapsed_time")
    if i > 0
        num_rows[i] = m
        run_time[i] = elapsed_time
    end
end

plot(num_rows, run_time, seriestype = :scatter, label = "LBFGS")
mkpath("./results/")
savefig("./results/lbfgs_scalability.png")
