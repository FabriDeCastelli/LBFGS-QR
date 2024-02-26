module LBFGS

using LinearAlgebra: norm, I, dot
using DataStructures: CircularBuffer
using ..OracleFunction

export LimitedMemoryBFGS

local const armijiowolfeorexact = :exact

function ArmijoWolfeLineSearch(
        f::Union{LeastSquaresF, OracleF},
        x::AbstractArray,
        p::AbstractArray,
        MaxEvaluations::Integer;
        αinit::Real=1,
        τ::Real=1.1,
        c1::Real=1e-4,
        c2::Real=0.9,
        ϵα::Real=1e-16,
        ϵgrad::Real=1e-12,
        safeguard::Real=0.20,
    )::Tuple{Real, Integer}

    ϕ = (α) -> begin
        v = f.eval(x + α * p)
        gradient = f.grad(x + α * p)
        return (v, dot(p, gradient))
    end

    α = αinit
    local αgrad

    ϕ_0, ϕd_0 = ϕ(0)

    while MaxEvaluations > 0
        αcurr, αgrad = ϕ(α)
        MaxEvaluations -= 2

        if (αcurr ≤ ϕ_0 + c1 * α * ϕd_0) && (abs(αgrad) ≤ -c2 * ϕd_0)
            return (α, MaxEvaluations)
        end
        
        if αgrad ≥ 0
            break
        end
        α *= τ
    end

    αlo = 0
    αhi = α
    αlograd = ϕd_0
    αhigrad = αgrad

    while (MaxEvaluations > 0) && (αhi - αlo) > ϵα && (αgrad > ϵgrad)
        α = (αlo * αhigrad - αhi * αlograd)/(αhigrad - αlograd)
        α = max(
            αlo + (αhi - αlo) * safeguard,
            min(αhi - (αhi - αlo) * safeguard, α)
        )

        αcurr, αgrad = ϕ(α)
        MaxEvaluations -= 2

        if (αcurr ≤ ϕ_0 + c1 * α * ϕd_0) && (abs(αgrad) ≤ -c2 * ϕd_0)
            break
        end

        if αgrad < 0
            αlo = α
            αlograd = αgrad
        else
            αhi = α
            if αhi ≤ ϵα
                break
            end
            αhigrad = αgrad
        end
    end

    return (α, MaxEvaluations)
end

function ExactLineSearch(
        f::LeastSquaresF,
        x::AbstractArray,
        p::AbstractArray,
        MaxEvaluations::Integer
    )
    MaxEvaluations -= 1
    return (tomography(f, x, p), MaxEvaluations)
end

@doc raw"""
﻿LimitedMemoryBFGS(f::Union{LeastSquaresF{T}, OracleF{T, F, G}}, [x::AbstractVector{T}, ϵ::T=1e-6, m::Integer=3, MaxEvaluations::Integer=10000])

Computes the minimum of the input function `f`.

### Input

- `f` -- the input function to minimize.
- `x` -- the starting point, if not specified the default one for the function `f` is used.
- `ϵ` -- the tollerance for the stopping criteria.
- `m` -- maximum number of vector to store that compute the approximate hessian.
- `MaxEvaluations` -- maximum number of function evaluations. Both ```f.eval``` and ```f.grad``` are counted.

### Output

A named tuple containing:
- `x` -- the minimum found
- `eval` -- the value of the function at the minimum
- `grad` -- the gradient of the function at the minimum
- `RemainingEvaluations` -- the number of function evaluation not used.

See also [`QRhous`](@ref).
"""
function LimitedMemoryBFGS(
        f::Union{LeastSquaresF{T}, OracleF{T, F, G}};
        x::Union{Nothing, AbstractVector{T}}=nothing,
        ϵ::T=1e-6,
        m::Integer=3,
        MaxEvaluations::Integer=10000
    )::Tuple{AbstractVector{T}, T, AbstractVector{T}, Integer} where {T, F<:Function, G<:Function}

    if isnothing(x)
        x = f.starting_point
    end

    k = 0
    gradient = f.grad(x)
    MaxEvaluations -= 1
    normgradient0 = norm(gradient)
    H = CircularBuffer{NamedTuple}(m)

    while MaxEvaluations > 0 && norm(gradient) > ϵ * normgradient0
        # two loop recursion for finding the direction
        q = gradient
        αstore = Array{eltype(x)}(undef, 0)

        for i ∈ reverse(H)
            αi = i[:ρ] * i[:s]' * q
            push!(αstore, αi)
            q -= αi * i[:y]
        end
        # choose H0 as something resembling the hessian
        H0 = if isempty(H)
            I
        else
            ((H[end][:s]' * H[end][:y])/(H[end][:y]' * H[end][:y])) * I
        end
        r = H0 * q
        for i ∈ H
            βi = i[:ρ] * i[:y]' * r
            r += i[:s] * (pop!(αstore) - βi)
        end
        p = -r # direction

        if armijiowolfeorexact === :armijiowolfe || f isa OracleF
            α, MaxEvaluations = ArmijoWolfeLineSearch(f, x, p, MaxEvaluations)
        elseif armijiowolfeorexact === :exact
            α, MaxEvaluations = ExactLineSearch(f, x, p, MaxEvaluations)
        end

        previousx = x
        x = x + α * p

        previousgradient = gradient
        gradient = f.grad(x)
        MaxEvaluations -= 1

        s = x - previousx
        y = gradient - previousgradient

        curvature = dot(s, y)
        ρ = 1 / curvature

        if curvature ≤ 1e-16
            empty!(H) # restart from the gradient
        else
            push!(H, (; :ρ => ρ, :y => y, :s => s))
        end
    end

    return (;
        :x => x,
        :eval => f.eval(x),
        :grad => gradient,
        :RemainingEvaluations => MaxEvaluations)
end

end # module LBGGS
