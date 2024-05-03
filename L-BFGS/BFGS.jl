module BFGS

using LinearAlgebra: norm, I, dot, diagm, mul!

using ..OracleFunction

export BroydenFletcherGoldfarbShanno, BroydenFletcherGoldfarbShannoDogleg

const armijiowolfeorexact = :exact
BFGSorDFP = :BFGS

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
```julia
BroydenFletcherGoldfarbShanno(f::Union{LeastSquaresF, OracleF}, [x::AbstractVector{T}, ϵ::T=1e-6, MaxEvaluations::Integer=10000])
```

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

"""
function BroydenFletcherGoldfarbShanno(
        f::Union{LeastSquaresF, OracleF};
        x::Union{Nothing, AbstractVector{T}}=nothing,
        ϵ::T=1e-6,
        MaxEvaluations::Integer=10000
    )::NamedTuple where {T}

    if isnothing(x)
        x = f.starting_point
    end

    gradient = f.grad(x)
    MaxEvaluations -= 1
    normgradient0 = norm(gradient)
    H = diagm(ones(length(x)))
    tmp1 = similar(H)
    tmp2 = similar(H)

    firstEvaluation = true

    while MaxEvaluations > 0 && norm(gradient) > ϵ * normgradient0
        p = -H * gradient # direction

        α, MaxEvaluations =
            if armijiowolfeorexact === :armijiowolfe || f isa OracleF
                ArmijoWolfeLineSearch(f, x, p, MaxEvaluations)
            elseif armijiowolfeorexact === :exact
                ExactLineSearch(f, x, p, MaxEvaluations)
            end

        previousx = x
        x = x + α * p

        previousgradient = gradient
        gradient = f.grad(x)
        MaxEvaluations -= 1

        s = x - previousx
        y = gradient - previousgradient
        ρ = inv(dot(y, s))

        # if its the first iteration then set H to an aproximation of the Hessian
        if firstEvaluation
            mul!(H, I, dot(y, s)/dot(y, y))
            firstEvaluation = false
        end

        if BFGSorDFP == :DFP
            # DFP update -------------------------------------------
            # H = H - (H * y * y' * H)/(y' * H * y) + (s * s')/(y' * s)

            mul!(tmp1, H * y * y', H)
            mul!(tmp2, s, s')
            H .+= -tmp1/dot(y, H, y) .+ ρ * tmp2
        elseif BFGSorDFP == :BFGS
            # BFGS update ------------------------------------------
            # H = (I - ρ * s * y') * H * (I - ρ * y * s') + ρ * s * s'

            mul!(tmp1, H * y, s')
            mul!(tmp2, s, s')
            H .+= ρ * ((1 + ρ * dot(y, H, y)) .* tmp2 .- tmp1 .- tmp1')
        end
    end

    return (;
        :x => x,
        :eval => f.eval(x),
        :grad => gradient,
        :RemainingEvaluations => MaxEvaluations)
end


function BroydenFletcherGoldfarbShannoDogleg(
        f::Union{LeastSquaresF, OracleF};
        x::Union{Nothing, AbstractVector{T}}=nothing,
        ϵ::T=1e-6,
        MaxEvaluations::Integer=10000
    )::NamedTuple where {T}

    if isnothing(x)
        x = f.starting_point
    end

    Δ = 1 # initial size of trust region
    smallestΔ = 1e-4 # smallest size where linear aproximation is applied

    gradient = f.grad(x)
    MaxEvaluations -= 1
    normgradient0 = norm(gradient)
    normgradient = normgradient0
    H = diagm(ones(length(x)))
    B = copy(H)
    tmp1 = similar(H)
    tmp2 = similar(H)
    tmp3 = similar(H)

    firstEvaluation = true

    while MaxEvaluations > 0 && norm(gradient) > ϵ * normgradient0
        # compute s by solving the subproblem min_s grad' * s + 0.5 s' * B * s with norm(s) ≤ Δ
        CauchyPoint = -(Δ/normgradient) * gradient
        τ = if dot(gradient, B, gradient) ≤ 0
                1
            else
                min((normgradient^3)/(Δ * dot(gradient, B, gradient)), 1)
            end

        if Δ ≤ smallestΔ || B == I
            # the Cauchy point is enought for small regions (linear aproximation)
            s = τ * CauchyPoint
        else
            pB = -H * gradient
            pU = -dot(gradient, gradient)/dot(gradient, B, gradient) * gradient

            if norm(pB) ≤ Δ
                # the region is larger than the dogleg
                s = pB
            elseif Δ ≤ norm(pU)
                # the region is smaller than the first step
                s = Δ/norm(pU) * pU
            else
                # solve the quadratic sistem for the dogleg
                one = dot(pU, (pB - pU))
                two = dot(pB - pU, pB - pU)
                three = dot(pU, pU)

                τ = (-one+two + sqrt(one^2 - three * two + two * Δ^2))/two
                s = pU + (τ - 1) * (pB - pU)
            end
        end

        previousx = x
        x = x + s

        previousgradient = gradient
        gradient = f.grad(x)
        normgradient = norm(gradient)
        MaxEvaluations -= 1

        y = gradient - previousgradient
        ρ = inv(dot(y, s))

        ared = f.eval(x) - f.eval(x + s) # actual reduction
        pred = -(dot(gradient, s) + 0.5 * dot(s, B, s)) # predicted reduction
        MaxEvaluations -= 2

        # expand or contract the region
        if (0.75 < ared/pred) && (0.8 * Δ < norm(s))
            Δ = 2 * Δ
        elseif (ared/pred < 0.1)
            Δ = 0.5 * Δ
        end

        # if its the first iteration then set H to an aproximation of the Hessian
        if firstEvaluation
            mul!(H, I, dot(y, s)/dot(y, y))
            firstEvaluation = false
        end

        if BFGSorDFP == :DFP
            # DFP update -------------------------------------------
            # H = H - (H * y * y' * H)/(y' * H * y) + (s * s')/(y' * s)

            mul!(tmp1, H * y * y', H)
            mul!(tmp2, s, s')
            H .+= -tmp1/dot(y, H, y) .+ ρ * tmp2

            mul!(tmp1, y, s')
            tmp2 = I - ρ * tmp1
            mul!(tmp1, tmp2, B)
            mul!(tmp3, tmp1, tmp2')
            mul!(tmp2, y, y')
            B .= tmp3 .+ ρ * tmp2
        elseif BFGSorDFP == :BFGS
            # BFGS update ------------------------------------------
            # H = (I - ρ * s * y') * H * (I - ρ * y * s') + ρ * s * s'

            mul!(tmp1, H * y, s')
            mul!(tmp2, s, s')
            H .+= ρ * ((1 + ρ * dot(y, H, y)) .* tmp2 .- tmp1 .- tmp1')

            mul!(tmp1, B * s * s', B)
            mul!(tmp2, y, y')
            B .+= -tmp1/dot(s, B, s) .+ ρ * tmp2
        end
    end

    return (;
        :x => x,
        :eval => f.eval(x),
        :grad => gradient,
        :RemainingEvaluations => MaxEvaluations)
end

end # module BFGS