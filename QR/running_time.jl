
include("housQR.jl")
include("../utilities/functions.jl")
using .housQR
using Plots

import Plots.plot

runs = 10
num_rows = zeros(runs)
run_time = zeros(runs)

warm_up()

for i in 0:runs
    m = 500 + i * 500
    A = rand(m, 300)
    b = rand(m)
    elapsed_time = @elapsed begin
        QR = qrfact(A)
    end 
    println("Time for $m rows: $elapsed_time")
    if i > 0
        num_rows[i] = m
        run_time[i] = elapsed_time
    end
    x = QR \ b
end

plot(num_rows, run_time, seriestype = :scatter, label = "QR Factorization")
savefig("results/qr_scalability.png")
