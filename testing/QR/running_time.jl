include("../../QR/housQR.jl")
using .housQR
using Plots: plot, savefig

runs = 5
num_rows = zeros(runs)
run_time = zeros(runs)

# warm up
qrfact([0. 1; 1 0]) \ [1.; 1]

for i ∈ 0:runs
    # m = 500 + i * 500
    m = 300
    #A = rand(m, 300)
    A = rand(m, m)
    b = rand(m)
    elapsed_time = @elapsed begin
        QR = qrfact(A)
        x = QR \ b
    end 
    println("Time for $m rows: $elapsed_time")
    if i > 0
        num_rows[i] = m
        run_time[i] = elapsed_time
    end
end

plot(num_rows, run_time, seriestype = :scatter, label = "QR Factorization")
mkpath("./results/")
savefig("./results/qr_scalability.png")
