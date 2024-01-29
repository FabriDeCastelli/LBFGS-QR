

function warm_up()
    println("Warm up the JIT compiler")
    for _ in 1:10
        rand(100,100) \ rand(100)
    end
end