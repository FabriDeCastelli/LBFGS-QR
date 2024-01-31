include("housQR.jl")
include("../utilities/dataset.jl")
using .housQR
using LinearAlgebra: norm, \, qr

lambdas = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

for lambda in lambdas
    X_hat, y_hat, _, w_star = get_dataset("../data_for_testing/dataset.csv", lambda)

    # ----------- Our QR factorization -----------
    QR = qrfact(X_hat)
    X = QR.Q * QR.R
    accuracy = norm(X_hat - X) / norm(X_hat)

    println("(Ours) Lambda = $lambda: relative error = $relative_error, accuracy = $accuracy")

    # ----------- Julia's QR factorization -----------
    Q, R = qr(X_hat)
    X = Q * R
    accuracy = norm(X_hat - X) / norm(X_hat)

    println("(Julia's) Lambda = $lambda: relative error = $relative_error, accuracy = $accuracy")
end
