using CSV
using DataFrames
using LinearAlgebra

@doc raw"""
    get_dataset(file_path::String, lambda::Float64)

Reads the ML-CUP-23/24 dataset from the given file path and returns the
necessary matrices and vectors to solve the problem. The LS problem is
solved with the standard solver in Julia.

### Input

- 'file_path' -- the path to the dataset file.
- 'lambda' -- the regularization parameter.

### Output

A tuple containing

- 'X_hat' -- the augmented matrix X.
- 'y_hat' -- the augmented vector y.
- 'w' -- the initial starting point.
- 'w_star' -- the optimal solution.

"""
function get_dataset(file_path::String, lambda::Float64)
    
    X = CSV.File(file_path; header=false) |> DataFrame
    X = Matrix(X)

    m, n0 = size(X)
    X_hat = [transpose(X); lambda .* I(m)]
    m, n = size(X_hat)
    y_hat = [randn(min(m, n0)); zeros(abs(m - n0))]


    # w is our initial starting point 
    w = ones(n)

    # w_star is the optimal solution of the julia solver 
    w_star = X_hat \ y_hat

    return X_hat, y_hat, w, w_star
end
