include("OracleFunction.jl")
include("LBFGS.jl")
include("../utilities/dataset.jl")


using ..LBFGS
using ..OracleFunction
using LinearAlgebra: norm


# lambdas = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
lambdas = [0.]

# warm up
# LimitedMemoryBFGS(LeastSquaresF([1., 1], [1. 1; 0 1], [2., -1]))

for lambda in lambdas
    X_hat, y_hat, w_inital, w_star = get_dataset("data_for_testing/dataset.csv", lambda)

    # ----------- OURS -----------
    w, fw, grad_fw = LimitedMemoryBFGS(LeastSquaresF(randn(size(X_hat, 2)), X_hat, y_hat), ϵ=1e-8, m=7, MaxEvaluations=10_000, x=w_inital)
    # w, fw, grad_fw = BFGS(LeastSquaresF(ones(size(X_hat, 2)), X_hat, y_hat), x=w_inital, ϵ=1e-8, MaxEvaluations=10_000)


    f = LeastSquaresF(randn(size(X_hat, 2)), X_hat, y_hat)
    relative_error = norm(w - w_star) / norm(w_star)

    println("------- λ = $lambda -------")
    println("||w||: $(norm(w))")
    println("||(julia solver) optimal w||: $(norm(w_star))")
    println("f(w_star): $(f.eval(w_star))")
    println("f(w): $fw")
    println("||grad(f(w))||: $(norm(grad_fw))")
    println("Relative error: $relative_error")
    println(" ")

end


