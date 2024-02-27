include("dataset.jl")
using Random

datasetpath = joinpath(@__DIR__, "../data_for_testing/dataset.csv")
VALID = [:dataset, :randDataset, :exactRandDataset, :rand, :exactRand]

@doc raw"""
﻿genFunc(t::Symbol, [lambda::Real=1, m::Integer=1000, n::Integer=10, rng=default_rng()])

Generates different matrices for the least squares problem

### Input

- `t` -- The type of problem to generate. Possible values are:
`:dataset`,
`:randDataset`,
`:exactRandDataset`,
`:rand`,
`:exactRand`

### Output

A named tuple with the matrix ``\hat{X}``, the vector ``\hat{y}``, the starting vector and the solution vector ``w^*``.

See also [`gen_dataset`](@ref).
"""
function genFunc(
        t::Symbol;
        λ::Real=1,
        m::Integer=1000,
        n::Integer=10,
        rng=Random.default_rng())::NamedTuple
    if t ∉ VALID
        throw(ArgumentError("The type of matrix to produce is not recognized, available types: " * join(String.(VALID), ", ")))
    end

    if t == :dataset
        tmp = get_dataset(datasetpath, λ)
        return (; :X_hat => tmp[1], :y_hat => tmp[2], :start => tmp[3], :w_star => tmp[4])
    end

    if t == :randDataset
        X_hat = [(rand(rng, n, m).*2 .-1); λ .* I(m)]
        y_hat = [(rand(rng, n).*2 .-1); zeros(m)]

        # initial starting point 
        start = ones(m)

        # w_star is the optimal solution of the julia solver 
        w_star = X_hat \ y_hat

        return (; :X_hat => X_hat, :y_hat => y_hat, :start => start, :w_star => w_star)
    end

    if t == :exactRandDataset
        X_hat = [(rand(rng, n, m).*2 .-1); λ .* I(m)]
        w_star = rand(rng, m).*2 .-1

        # initial starting point 
        start = ones(m)

        y_hat = X_hat * w_star

        return (; :X_hat => X_hat, :y_hat => y_hat, :start => start, :w_star => w_star)
    end

    if t == :rand
        X_hat = rand(rng, m+n, m).*2 .-1
        y_hat = rand(rng, m+n)

        # initial starting point 
        start = ones(m)

        # w_star is the optimal solution of the julia solver 
        w_star = X_hat \ y_hat

        return (; :X_hat => X_hat, :y_hat => y_hat, :start => start, :w_star => w_star)
    end

    if t == :exactRand
        X_hat = rand(rng, m+n, m).*2 .-1
        w_star = rand(rng, m).*2 .-1

        # initial starting point 
        start = ones(m)

        y_hat = X_hat * w_star

        return (; :X_hat => X_hat, :y_hat => y_hat, :start => start, :w_star => w_star)
    end
end
